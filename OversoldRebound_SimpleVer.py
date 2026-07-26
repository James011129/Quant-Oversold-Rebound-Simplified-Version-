# 聚宽策略
# 沪深主板 - 下跌趋势止跌回升策略 (无未来函数 + 严格 $T-1$ 数据对齐)
from jqdata import *
import pandas as pd
import numpy as np
import talib

def initialize(context):
    # 沪深300作为benchmark
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)

    # 交易费率设置
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.0005,
            open_commission=0.0001,
            close_commission=0.0001,
            min_commission=0.3
        ),
        type='stock'
    )

    # 核心参数
    g.max_hold = 2        # 最大持仓数量
    g.min_money = 1e8     # 最低成交额1亿
    g.volume_ratio = 1.5  # 量比阈值（用于确认相对量能放大）
    
    # 定义仓位分配权重
    g.position_weights = [0.7, 0.3]

    # 运行频率设定：每日开盘运行
    run_daily(trade, time='open')

    log.info("策略初始化完成。最大持仓: %d只，策略模式：无未来函数 - 下跌止跌回升。" % g.max_hold)


# ======================
# 交易主逻辑
# ======================

def trade(context):
    log.info("=" * 20 + f" {context.current_dt.strftime('%Y-%m-%d')} 交易逻辑开始 " + "=" * 20)

    # ==================
    # 选股模块
    # ==================
    candidate_stocks = check_stocks(context)
    
    # 输出前五只股票的详细信息（如果选出的股票不足5只则全部展示）
    if candidate_stocks:
        log.info("【选股结果】今日依据 T-1 日数据得分排名前5的标的：")
        log.info("-" * 100)
        log.info(f"{'排名':<6}{'代码':<12}{'名称':<10}{'得分':<10}{'换手率(%)':<12}{'昨日量比':<10}{'昨日涨幅(%)':<12}")
        log.info("-" * 100)
        
        for i, stock_info in enumerate(candidate_stocks[:5], 1):
            log.info(
                f"{i:<6}{stock_info['code']:<12}{stock_info['name']:<10}"
                f"{stock_info['score']:<10.2f}{stock_info['turnover_ratio']:<12.2f}"
                f"{stock_info['volume_ratio']:<10.2f}{stock_info['daily_return']:<12.2f}"
            )
        
        log.info("-" * 100)
    else:
        log.info("【选股结果】今日无符合条件的标的。")
        
    # 获取总资产用于计算目标仓位金额
    total_value = context.portfolio.total_value
    
    # 按设定仓位买入
    target_positions = {}
    for i, stock_info in enumerate(candidate_stocks[:g.max_hold]):
        target_positions[stock_info['code']] = g.position_weights[i]

    # ==================
    # 调仓与卖出模块
    # ==================
    # 自动调仓：卖出不在最新 target_positions 中的持仓
    for stock in list(context.portfolio.positions.keys()):
        if stock not in target_positions:
            log.info(f"🟡【平仓】发起对 {stock} 的全额平仓委托。")
            order_target_value(stock, 0)

    # ==================
    # 买入与资金分配模块
    # ==================
    for stock, weight in target_positions.items():
        target_cash = total_value * weight
        log.info(f"🔵【买入】标的 {stock}，目标仓位占比：{weight*100}%，目标金额：{target_cash:.2f}")
        order_target_value(stock, target_cash)


# ======================
# 股票筛选与评分
# ======================

def check_stocks(context):
    current = get_current_data()
    # 严格使用上一个交易日日期获取基本面数据，绝对避免使用 context.current_dt
    prev_date = context.previous_date

    # 获取当前交易日可交易的全部 A 股列表
    all_stocks = get_all_securities(['stock'], date=context.current_dt).index.tolist()

    # 1. 第一层过滤：沪深主板及停牌过滤
    target_list = []
    for stock in all_stocks:
        code = stock[:3]
        # 只做沪深主板
        # if not (code.startswith('600') or code.startswith('601') or
        #         code.startswith('603') or code.startswith('605') or
        #         code.startswith('000') or code.startswith('001')):
        #     continue
        # # 排除创业板/科创板/北交所
        # if stock.startswith('300') or stock.startswith('688') or stock.startswith('8'):
        #     continue
        # 实时停牌过滤
        if current[stock].paused:
            continue

        target_list.append(stock)

    if not target_list:
        return []

    # 2. 第二层过滤：基于 T-1 日已结算换手率筛选 (3% ~ 14%)
    q = query(valuation.code, valuation.turnover_ratio).filter(
        valuation.code.in_(target_list),
        valuation.turnover_ratio >= 3,
        valuation.turnover_ratio <= 14
    )
    df_val = get_fundamentals(q, date=prev_date)
    
    if df_val is None or df_val.empty:
        return []

    # 将换手率预先存入字典映射，避免在循环中重复查询（无未来函数且极大提升运行效率）
    turnover_map = dict(zip(df_val['code'], df_val['turnover_ratio']))
    valid_stocks = df_val['code'].tolist()

    candidate_stocks = []

    # 3. 技术指标与历史行情校验 (以 T-1 日为最新收盘日)
    for stock in valid_stocks:
        try:
            # 此时开盘获取的 40 根 K 线，最后一根 iloc[-1] 严格对应 T-1 日收盘
            hist = attribute_history(
                stock, 40, '1d',
                ['close', 'high', 'low', 'volume', 'money'],
                skip_paused=True, df=True
            )

            if len(hist) < 35:
                continue

            close = hist['close']
            volume = hist['volume']
            
            # 提取 NumPy 数组用于 talib 计算
            close_vals = close.values
            high_vals = hist['high'].values
            low_vals = hist['low'].values

            # T-1 日成交额过滤 > 1亿
            if hist['money'].iloc[-1] < g.min_money:
                continue

            # ===========================
            # 核心逻辑 1：定义“下跌趋势”(基于 T-1 日及更早数据)
            # ===========================
            # roc_20 = (close.iloc[-16] / close.iloc[-21]) - 1
            roc_15 = (close.iloc[-11] / close.iloc[-16]) - 1
            roc_10 = (close.iloc[-6] / close.iloc[-11]) - 1
            roc_5 = (close.iloc[-1] / close.iloc[-6]) - 1
            ma10_y = close.iloc[-11:-1].mean()
            ma10_5d_ago = close.iloc[-16:-6].mean()

            if  roc_15 >= 0 or roc_10 >= 0 or roc_5 >= 0 or ma10_y >= ma10_5d_ago:
                continue 

            # ===========================
            # 核心逻辑 2：定义“止跌回升”(T-1日开启上涨，T-2日依旧下跌)
            # ===========================
            if not (close.iloc[-1] >= close.iloc[-2] and close.iloc[-2] <= close.iloc[-3]):
                continue

            # ===========================
            # 核心逻辑 3：定义“量能放大”
            # ===========================
            if volume.iloc[-1] <= volume.iloc[-2]:
                continue

            avg_volume = volume.iloc[-6:-1].mean()
            if avg_volume <= 0:
                continue
            ratio = volume.iloc[-1] / avg_volume
            if ratio < g.volume_ratio:
                continue

            # ===========================
            # 核心逻辑 4：多因子指标评分系统
            # ===========================
            indicator_score = 0
            
            # (1) MA5 金叉 MA10
            ma5 = talib.SMA(close_vals, timeperiod=5)
            ma10 = talib.SMA(close_vals, timeperiod=10)
            if ma5[-2] <= ma10[-2] and ma5[-1] > ma10[-1]:
                indicator_score += 30
                
            # (2) MACD 金叉
            macd_dif, macd_dea, macd_hist = talib.MACD(close_vals, fastperiod=12, slowperiod=26, signalperiod=9)
            if macd_dif[-2] <= macd_dea[-2] and macd_dif[-1] > macd_dea[-1]:
                indicator_score += 25
                
            # (3) KDJ 多头/金叉
            slowk, slowd = talib.STOCH(high_vals, low_vals, close_vals, fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            if slowk[-2] <= slowd[-2] and slowk[-1] > slowd[-1]:
                indicator_score += 20
                
            # (4) RSI 多头区间 (50 < RSI14 < 70)
            rsi = talib.RSI(close_vals, timeperiod=14)
            if 70 > rsi[-1] > 50:
                indicator_score += 20
                
            # (5) BOLL 多头 (收盘价站上中轨)
            upper, middle, lower = talib.BBANDS(close_vals, timeperiod=20)
            if close_vals[-1] > middle[-1]:
                indicator_score += 10

            # 基础强度得分：(T-1日涨幅百分比) + (T-1日量比)
            daily_return_pct = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100
            base_strength = daily_return_pct + ratio

            # 总分 = 技术指标多头得分 + 当日爆发基础分 * 10
            total_score = indicator_score + base_strength 
            
            # 获取股票名称
            stock_info = get_security_info(stock)
            stock_name = stock_info.display_name if stock_info else "未知名称"
            
            # 从此前批量查好的映射表中获取 T-1 日换手率
            turnover_ratio = turnover_map.get(stock, 0.0)

            candidate_stocks.append({
                'code': stock,
                'name': stock_name,
                'score': total_score,
                'turnover_ratio': turnover_ratio,
                'volume_ratio': ratio,
                'daily_return': daily_return_pct
            })

        except Exception as e:
            continue

    # 4. 按总评分降序排列
    candidate_stocks.sort(key=lambda x: x['score'], reverse=True)

    # 5. 返回股票列表
    return candidate_stocks
