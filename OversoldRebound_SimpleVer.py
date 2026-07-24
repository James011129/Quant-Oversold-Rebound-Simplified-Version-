# 聚宽策略
# 沪深主板 - 下跌趋势止跌回升策略 (带详尽日志)
from jqdata import *
import pandas as pd
import numpy as np


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
    g.max_hold = 3  # 最大持仓数量
    g.min_money = 1e8  # 最低成交额2亿
    g.volume_ratio = 1.5  # 量比阈值（用于确认相对量能放大）

    # 运行频率设定
    run_daily(trade, time='open')

    log.info("策略初始化完成。最大持仓: %d只，策略模式：下跌趋势止跌回升 + 放量金叉。" % g.max_hold)


# ======================
# 交易主逻辑
# ======================

def trade(context):
    log.info("=" * 20 + " 今日交易逻辑开始 " + "=" * 20)

    # ==================
    # 选股与调仓模块
    # ==================
    buy_list = check_stocks(context)

    if buy_list:
        log.info(f"【选股结果】今日符合条件的标的：{buy_list}")
    else:
        log.info("【选股结果】今日无符合条件的标的。")

    # 自动调仓：卖出不在最新 buy_list 中的持仓
    for stock in list(context.portfolio.positions.keys()):
        log.info(f"🟡【次日平仓】执行隔日卖出策略，发起对 {stock} 的平仓委托。")
        order_target(stock, 0)

    # 当前持仓统计
    hold_num = len(context.portfolio.positions)
    if hold_num >= g.max_hold:
        log.info("【仓位已满】当前持仓达到最大数量，今日无可用资金建仓。")
        return

    need_buy = g.max_hold - hold_num
    cash = context.portfolio.cash / need_buy if need_buy > 0 else 0

    # 买入新股
    for stock in buy_list:
        if stock not in context.portfolio.positions:
            log.info(f"🔵【建仓买入】买入 {stock}，计划分配资金：{cash:.2f}")
            order_value(stock, cash)
            if len(context.portfolio.positions) >= g.max_hold:
                break


# ======================
# 股票筛选
# ======================

def check_stocks(context):
    current = get_current_data()
    date = context.current_dt

    # 全部A股
    all_stocks = get_all_securities(['stock'], date=date).index.tolist()

    # 1. 第一层过滤：沪深主板及停牌过滤
    target_list = []
    for stock in all_stocks:
        code = stock[:3]
        # 只做沪深主板
        if not (code.startswith('600') or code.startswith('601') or
                code.startswith('603') or code.startswith('605') or
                code.startswith('000') or code.startswith('001')):
            continue
        # 排除创业板/科创板/北交所
        if stock.startswith('300') or stock.startswith('688') or stock.startswith('8'):
            continue
        # 停牌及ST过滤
        if current[stock].paused :
            continue

        target_list.append(stock)

    if not target_list:
        return []

    # 2. 第二层过滤：换手率
    q = query(valuation.code, valuation.turnover_ratio).filter(
        valuation.code.in_(target_list),
        valuation.turnover_ratio >= 3,
        valuation.turnover_ratio <= 10
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

            # 成交额过滤 > 2亿
            if hist['money'].iloc[-1] < g.min_money:
                continue

            # ===========================
            # 核心逻辑 1：定义“下跌趋势”
            # ===========================
            # 条件A: 整体收盘价相较于20天前是下跌的
            roc_20 = (close.iloc[-16] / close.iloc[-21]) - 1
            roc_15 = (close.iloc[-11] / close.iloc[-16]) - 1
            roc_10 = (close.iloc[-6] / close.iloc[-11]) - 1
            roc_5 = (close.iloc[-1] / close.iloc[-6]) - 1
            # 条件B: 20日均线整体呈下降趋势 (昨日MA20 < 5日前的MA20)
            ma15_y = close.iloc[-21:-1].mean()
            ma15_5d_ago = close.iloc[-26:-6].mean()

            if roc_20 >= 0 or roc_15>=0 or roc_10>=0 or roc_5>=0 or ma15_y >= ma15_5d_ago:
                continue  # 不属于下跌趋势，直接过滤

            # ===========================
            # 核心逻辑 2：定义“止跌回升” (MA5金叉MA10)
            # ===========================
            # ma5 = close.iloc[-5:].mean()
            # ma10 = close.iloc[-10:].mean()
            # ma5_y = close.iloc[-6:-1].mean()
            # ma10_y = close.iloc[-11:-1].mean()
            #
            # if not (ma5_y <= ma10_y and ma5 > ma10):
            #     continue

            #昨日开启上涨,前日依旧下跌
            if not (close.iloc[-1] >=close.iloc[-2] and close.iloc[-2] <= close.iloc[-3]):
                continue

            # ===========================
            # 核心逻辑 3：定义“量能放大”
            # ===========================
            # 绝对放大：今日成交量严格大于昨日
            if volume.iloc[-1] <= volume.iloc[-2]:
                continue

            # 相对放大：量比过滤
            avg_volume = volume.iloc[-6:-1].mean()
            if avg_volume <= 0:
                continue
            ratio = volume.iloc[-1] / avg_volume
            if ratio < g.volume_ratio:
                continue

            # ===========================
            # 核心逻辑 4：评分排序系统
            # ===========================
            # 评分公式：(当日涨幅百分比) + (当日量比)
            # 逻辑说明：回升的第一根阳线力度越强、资金介入程度（量比）越高，评分越高
            daily_return_pct = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100
            score = daily_return_pct + ratio

            candidate_scores.append((stock, score))

        except Exception as e:
            continue

    # 4. 评分降序排列
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    # 5. 提取股票返回
    sorted_candidates = [x[0] for x in candidate_scores]
    return sorted_candidates[:g.max_hold]