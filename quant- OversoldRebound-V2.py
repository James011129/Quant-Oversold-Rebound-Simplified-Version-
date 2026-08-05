# 聚宽策略
# 沪深主板 - 下跌趋势止跌回升策略 (增加技术多头评分 + 仓位管理)
from jqdata import *
import pandas as pd
import numpy as np
import talib


def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)

    # 交易费率设置
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.0005,
            open_commission=0.0001,
            close_commission=0.0001,
            min_commission=5
        ),
        type='stock'
    )

    # 核心参数
    g.max_hold = 2  # 最大持仓数量修改为 3 只[cite: 2]
    g.min_money = 1e8  # 最低成交额1亿[cite: 2]
    g.volume_ratio = 1.5  # 量比阈值（用于确认相对量能放大）[cite: 2]

    # 定义仓位分配权重 (5:3:2)
    g.position_weights = [0.7, 0.3]

    # 运行频率设定
    run_daily(trade, time='open')

    log.info("策略初始化完成。最大持仓: %d只，策略模式：下跌止跌回升 + 多因子评分。" % g.max_hold)


# ======================
# 交易主逻辑
# ======================

def trade(context):
    log.info("=" * 20 + " 今日交易逻辑开始 " + "=" * 20)

    # ==================
    # 选股模块
    # ==================
    buy_list = check_stocks(context)

    if buy_list:
        log.info(f"【选股结果】今日得分排名前2的标的：{buy_list}")
    else:
        log.info("【选股结果】今日无符合条件的标的。")

    # 获取总资产用于计算目标仓位金额
    total_value = context.portfolio.total_value

    # 建立今日目标持仓及对应权重的映射
    target_positions = {}
    for i, stock in enumerate(buy_list):
        if i < len(g.position_weights):
            target_positions[stock] = g.position_weights[i]

    # ==================
    # 调仓与卖出模块
    # ==================
    # 自动调仓：卖出不在最新 target_positions 中的持仓 (即次日平仓逻辑)
    for stock in list(context.portfolio.positions.keys()):
        if stock not in target_positions:
            log.info(f"🟡【次日平仓】发起对 {stock} 的全额平仓委托。")
            order_target_value(stock, 0)

    # ==================
    # 买入与资金分配模块
    # ==================
    # 按照 5:3:2 的目标权重调整对应股票的仓位
    for stock, weight in target_positions.items():
        target_cash = total_value * weight
        log.info(f"🔵【调仓买入】标的 {stock}，目标仓位占比：{weight * 100}%，目标金额：{target_cash:.2f}")
        order_target_value(stock, target_cash)


# ======================
# 股票筛选与评分
# ======================

def check_stocks(context):
    current = get_current_data()
    date = context.current_dt

    # 全部A股
    all_stocks = get_all_securities(['stock'], date=date).index.tolist()

    # 1. 第一层过滤：沪深主板及停牌过滤[cite: 2]
    target_list = []
    for stock in all_stocks:
        code = stock[:3]
        # 只做沪深主板[cite: 2]
        # if not (code.startswith('600') or code.startswith('601') or
        #         code.startswith('603') or code.startswith('605') or
        #         code.startswith('000') or code.startswith('001')):
        #     continue
        # 排除创业板/科创板/北交所[cite: 2]
        if stock.startswith('30') or stock.startswith('68') or stock.startswith('9'):
            continue
        # 停牌过滤[cite: 2]
        if current[stock].paused:
            continue

        target_list.append(stock)

    if not target_list:
        return []

    # 2. 第二层过滤：换手率 3~10%[cite: 2]
    q = query(valuation.code, valuation.turnover_ratio).filter(
        valuation.code.in_(target_list),
        valuation.turnover_ratio >= 3,
        valuation.turnover_ratio <= 15
    )
    df_val = get_fundamentals(q, date=date)
    valid_stocks = df_val['code'].tolist()

    candidate_scores = []

    # 3. 技术指标与历史行情校验
    for stock in valid_stocks:
        try:
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

            # 成交额过滤 > 1亿[cite: 2]
            if hist['money'].iloc[-1] < g.min_money:
                continue

            # ===========================
            # 核心逻辑 1：定义“下跌趋势”[cite: 2]
            # ===========================
            roc_20 = (close.iloc[-16] / close.iloc[-21]) - 1
            roc_15 = (close.iloc[-11] / close.iloc[-16]) - 1
            roc_10 = (close.iloc[-6] / close.iloc[-11]) - 1
            roc_5 = (close.iloc[-1] / close.iloc[-6]) - 1
            ma10_y = close.iloc[-11:-1].mean()
            ma10_5d_ago = close.iloc[-16:-6].mean()

            if roc_20 >= 0 or roc_15 >= 0 or roc_10 >= 0 or roc_5 >= 0 or ma10_y >= ma10_5d_ago:
                continue

                # ===========================
            # 核心逻辑 2：定义“止跌回升”[cite: 2]
            # ===========================
            # 昨日开启上涨,前日依旧下跌[cite: 2]
            if not (close.iloc[-1] >= close.iloc[-2] and close.iloc[-2] <= close.iloc[-3]):
                continue

            # ===========================
            # 核心逻辑 3：定义“量能放大”[cite: 2]
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

            # (1) MA5 金叉 MA10 (权重 20分)
            ma5 = talib.SMA(close_vals, timeperiod=5)
            ma10 = talib.SMA(close_vals, timeperiod=10)
            if ma5[-2] <= ma10[-2] and ma5[-1] > ma10[-1]:
                indicator_score += 40

            # (2) MACD 金叉 (权重 20分)
            macd_dif, macd_dea, macd_hist = talib.MACD(close_vals, fastperiod=12, slowperiod=26, signalperiod=9)
            if macd_dif[-2] <= macd_dea[-2] and macd_dif[-1] > macd_dea[-1]:
                indicator_score += 30

            # (3) KDJ 多头/金叉 (权重 20分)
            slowk, slowd = talib.STOCH(high_vals, low_vals, close_vals, fastk_period=9, slowk_period=3, slowk_matype=0,
                                       slowd_period=3, slowd_matype=0)
            if slowk[-2] <= slowd[-2] and slowk[-1] > slowd[-1]:
                indicator_score += 20

            # (4) RSI 多头区间 (权重 20分，RSI14 > 50视为多头)
            rsi = talib.RSI(close_vals, timeperiod=14)
            if rsi[-1] > 50:
                indicator_score += 20

            # (5) BOLL 多头 (权重 20分，收盘价站上中轨)
            upper, middle, lower = talib.BBANDS(close_vals, timeperiod=20)
            if close_vals[-1] > middle[-1]:
                indicator_score += 10

            # 基础强度得分：(当日涨幅百分比) + (当日量比)[cite: 2]
            daily_return_pct = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100
            base_strength = daily_return_pct + ratio

            # 总分 = 技术指标多头得分(满分100) + 当日爆发基础分
            total_score = indicator_score + base_strength * 8

            candidate_scores.append((stock, total_score))

        except Exception as e:
            continue

    # 4. 按总评分降序排列[cite: 2]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    sorted_candidates = [x[0] for x in candidate_scores]
    return sorted_candidates[:g.max_hold]