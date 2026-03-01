import akshare as ak
import pandas as pd
import numpy as np
import datetime
from scipy.stats import linregress
import time

# ==============================================================================
# 🎯 全局网格参数设置区 (可统一在此处修改)
# ==============================================================================
MIN_YEARS = 3                  # 最少上市年限 (淘汰次新股，规避新股泡沫破裂风险)
MIN_AVG_TURNOVER = 200000000   # 近20日均成交额门槛 (单位:元)。2亿！防流动性枯竭和庄股
MIN_DAILY_AMPLITUDE = 3.0      # 最低平均日振幅 (%)。股票有印花税，必须大于 3% 才有肉吃
MIN_CHOPPINESS = 0.90          # 最低震荡纯度 (0-1)。要求极度震荡，不走单边
MAX_TREND_SLOPE = 0.05         # 最大长期趋势斜率。剔除处于可怕的长期单边下跌通道的股票
MAX_SCAN_COUNT = 5500          # 最大扫描数量 (A股总数约5300只，可改小用于快速测试)
# ==============================================================================

def get_stock_list_safe():
    """获取全市场A股股票列表（避开东方财富接口）"""
    print("正在获取全市场A股列表...")
    try:
        # 使用基础接口获取股票代码字典，不受复杂封锁限制
        df = ak.stock_info_a_code_name()
        if 'code' in df.columns:
            df.rename(columns={'code': '代码', 'name': '名称'}, inplace=True)
        return df
    except Exception as e:
        print(f"获取股票列表失败，请检查网络: {e}")
        return pd.DataFrame()

def get_stock_hist_sina_safe(code, start_str, end_str):
    """获取个股历史数据（纯新浪接口，带前复权）"""
    code_str = str(code).zfill(6)
    
    # 构建新浪标准的 symbol: 沪市sh，深市sz。忽略北交所(8/4开头)防流动性陷阱
    if code_str.startswith(('6', '9')):
        symbol = f"sh{code_str}"
    elif code_str.startswith(('0', '3')):
        symbol = f"sz{code_str}"
    else:
        return None 

    try:
        # 新浪A股日K接口 (获取前复权数据 qfq)
        hist_df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_str, end_date=end_str, adjust="qfq")
        
        if hist_df is not None and not hist_df.empty:
            hist_df.rename(columns={'date':'日期', 'open':'开盘', 'high':'最高', 'low':'最低', 'close':'收盘', 'volume':'成交量'}, inplace=True)
            for col in['开盘', '收盘', '最高', '最低', '成交量']:
                hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
                
            hist_df.dropna(subset=['收盘', '成交量'], inplace=True)
            
            # 日期过滤
            hist_df['日期'] = pd.to_datetime(hist_df['日期']).dt.strftime('%Y%m%d')
            hist_df = hist_df[(hist_df['日期'] >= start_str) & (hist_df['日期'] <= end_str)].reset_index(drop=True)
            return hist_df
    except:
        pass
    return None

def calculate_stock_grid_metrics(df):
    """计算个股网格指标 (引入硬核常数过滤)"""
    df = df.sort_values('日期').reset_index(drop=True)
    
    # --- 流动性核查 ---
    # 新浪成交量单位通常是“股”。成交额 = 成交量 * 收盘价
    df['成交额'] = df['成交量'] * df['收盘']
    recent_20_days = df.tail(20)
    if len(recent_20_days) < 20: return None
        
    avg_turnover = recent_20_days['成交额'].mean()
    if avg_turnover < MIN_AVG_TURNOVER:
        return None
        
    # --- 计算 MA120 及趋势斜率 ---
    df['MA120'] = df['收盘'].rolling(window=120).mean()
    df = df.dropna().reset_index(drop=True)
    if len(df) < 120: return None
        
    ma120_cv = df['MA120'].std() / df['MA120'].mean()
    x = np.arange(len(df))
    y = df['MA120'].values
    slope, _, _, _, _ = linregress(x, y)
    trend_slope_pct = abs(slope) / df['MA120'].mean() * 100 
    
    # 剔除单边暴涨暴跌股
    if trend_slope_pct > MAX_TREND_SLOPE:
        return None

    # --- 振幅核查 ---
    df['前收盘'] = df['收盘'].shift(1)
    df.loc[0, '前收盘'] = df.loc[0, '开盘']
    df['日振幅'] = (df['最高'] - df['最低']) / df['前收盘']
    avg_daily_amplitude = df['日振幅'].mean() * 100 
    
    # 如果振幅过小，直接淘汰
    if avg_daily_amplitude < MIN_DAILY_AMPLITUDE:
        return None
        
    # --- 震荡纯度 (Choppiness) ---
    path_length = abs(df['收盘'] - df['前收盘']).sum()
    net_displacement = abs(df['收盘'].iloc[-1] - df['收盘'].iloc[0])
    choppiness = 1.0 - (net_displacement / (path_length + 0.0001))
    
    if choppiness < MIN_CHOPPINESS:
        return None
    
    # --- 综合评分 ---
    penalty = ma120_cv + trend_slope_pct
    grid_score = (avg_daily_amplitude * choppiness) / (penalty + 0.1)
    
    return {
        '网格评分': round(grid_score, 2),
        '日均振幅(%)': round(avg_daily_amplitude, 2),
        '震荡纯度': round(choppiness, 3),
        '近20日成交额': f"{avg_turnover / 100000000:.2f} 亿", 
        'MA变异系数': round(ma120_cv, 4),
        '趋势斜率': round(trend_slope_pct, 4)
    }

def scan_stocks_for_grid():
    stock_list = get_stock_list_safe()
    if stock_list.empty: return
        
    print(f"\n找到 {len(stock_list)} 只A股。即将开启地狱级条件筛选...")
    print(f"参数: 振幅>{MIN_DAILY_AMPLITUDE}%, 成交额>{MIN_AVG_TURNOVER/100000000}亿, 震荡度>{MIN_CHOPPINESS}\n")
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=MIN_YEARS * 365)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    results =[]
    count = 0
    valid_count = 0
    
    for index, row in stock_list.iterrows():
        count += 1
        if count > MAX_SCAN_COUNT: break
            
        code = str(row['代码']).strip()
        name = str(row['名称']).strip()
        
        # 坚决不碰 ST、退市股、北交所(8/4开头)
        if any(keyword in name for keyword in ["ST", "退"]) or code.startswith(('8', '4')):
            continue
            
        hist_df = get_stock_hist_sina_safe(code, start_str, end_str)
        
        if hist_df is not None and len(hist_df) >= (MIN_YEARS * 240) * 0.9:
            metrics = calculate_stock_grid_metrics(hist_df)
            
            if metrics:
                metrics['代码'] = code
                metrics['名称'] = name
                results.append(metrics)
                valid_count += 1
        
        if count % 100 == 0:
            print(f"进度：已扫描 {count} 只个股，当前通过“炼蛊”幸存标的：{valid_count} 只...")
            
        time.sleep(0.01) # 微小延迟，新浪接口比较抗造

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("\n扫描结束：要求过高，全市场无一只股票符合条件！你可以尝试调低开头的参数。")
        return None
        
    res_df = res_df.sort_values('网格评分', ascending=False).reset_index(drop=True)
    cols =['代码', '名称', '网格评分', '日均振幅(%)', '震荡纯度', '近20日成交额', 'MA变异系数', '趋势斜率']
    final_df = res_df[cols]
    
    print("\n================== 🎯 扫描完成！全市场最适合网格的“渣男”股票 TOP 15 ==================")
    print(final_df.head(15).to_string(index=False))
    
    final_df.to_csv("A股个股网格标的终极筛选表.csv", index=False, encoding="utf-8-sig")
    print("\n所有符合条件的标的已保存至：A股个股网格标的终极筛选表.csv")
    
    return final_df

if __name__ == "__main__":
    scan_stocks_for_grid()