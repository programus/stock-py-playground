import akshare as ak
import pandas as pd
import numpy as np
import datetime
from scipy.stats import linregress
import time

def get_etf_list_sina():
    """仅使用新浪接口获取全市场ETF列表"""
    print("正在连接新浪财经接口获取全市场ETF列表...")
    try:
        df = ak.fund_etf_category_sina(symbol="ETF基金")
        if 'symbol' in df.columns:
            df.rename(columns={'symbol': '代码', 'name': '名称'}, inplace=True)
        return df
    except Exception as e:
        print(f"获取ETF列表失败，请检查网络: {e}")
        return pd.DataFrame()

def get_etf_hist_sina_safe(code, start_str, end_str):
    """获取历史数据，修复了前缀识别逻辑，纯新浪源"""
    # 【应用了你修复的逻辑】：识别是否自带 sh/sz 前缀
    sina_symbol = code if str(code).startswith('s') else (f"sh{code}" if str(code).startswith(('5', '7')) else f"sz{code}")
    
    try:
        hist_df = ak.fund_etf_hist_sina(symbol=sina_symbol)
        
        if hist_df is not None and not hist_df.empty:
            hist_df.rename(columns={'date':'日期', 'open':'开盘', 'high':'最高', 'low':'最低', 'close':'收盘', 'volume':'成交量'}, inplace=True)
            # 强制转化为数值格式
            for col in['开盘', '收盘', '最高', '最低', '成交量']:
                hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
                
            hist_df.dropna(subset=['收盘', '成交量'], inplace=True)
            
            # 过滤时间范围
            hist_df['日期'] = pd.to_datetime(hist_df['日期']).dt.strftime('%Y%m%d')
            hist_df = hist_df[(hist_df['日期'] >= start_str) & (hist_df['日期'] <= end_str)].reset_index(drop=True)
            
            return hist_df
    except Exception as e:
        pass
        
    return None

def calculate_grid_metrics(df):
    """计算网格指标，加入流动性与振幅双重硬性过滤"""
    df = df.sort_values('日期').reset_index(drop=True)
    
    # ------------------ 一票否决 1：流动性过滤 ------------------
    # 计算每日成交额 (新浪的成交量单位通常为股，成交额 = 成交量 * 收盘价)
    df['成交额'] = df['成交量'] * df['收盘']
    
    # 取最近 20 个交易日的数据测算近期真实流动性
    recent_20_days = df.tail(20)
    if len(recent_20_days) < 20:
        return None
        
    avg_turnover = recent_20_days['成交额'].mean()
    
    # 【核心条件】：近20日平均日成交额必须 > 5000万 (50,000,000元)
    # 这也侧面保证了基金规模(AUM)绝大部分在几亿以上，无清盘风险
    if avg_turnover < 50000000:
        return None
    # ------------------------------------------------------------
    
    # 计算 MA120
    df['MA120'] = df['收盘'].rolling(window=120).mean()
    df = df.dropna().reset_index(drop=True)
    if len(df) < 120: return None
        
    # ------------------ 一票否决 2：振幅过滤 --------------------
    df['前收盘'] = df['收盘'].shift(1)
    df.loc[0, '前收盘'] = df.loc[0, '开盘']
    df['日振幅'] = (df['最高'] - df['最低']) / df['前收盘']
    avg_daily_amplitude = df['日振幅'].mean() * 100 
    
    # 【核心条件】：如果平均日振幅低于 1.5%，不够网格差价，淘汰！
    if avg_daily_amplitude < 1.5:
        return None
    # ------------------------------------------------------------

    # --- 长期稳定性 ---
    ma120_cv = df['MA120'].std() / df['MA120'].mean()
    x = np.arange(len(df))
    y = df['MA120'].values
    slope, _, _, _, _ = linregress(x, y)
    trend_slope_pct = abs(slope) / df['MA120'].mean() * 100 
    
    # --- 震荡纯度 (Choppiness Index) ---
    path_length = abs(df['收盘'] - df['前收盘']).sum()
    net_displacement = abs(df['收盘'].iloc[-1] - df['收盘'].iloc[0])
    choppiness = 1.0 - (net_displacement / (path_length + 0.0001))
    
    # --- 综合评分 V2 ---
    penalty = ma120_cv + trend_slope_pct
    grid_score = (avg_daily_amplitude * choppiness) / (penalty + 0.1)
    
    return {
        '网格综合评分': round(grid_score, 2),
        '平均日振幅(%)': round(avg_daily_amplitude, 2),
        '震荡纯度(0-1)': round(choppiness, 3),
        '近20日均成交额': f"{avg_turnover / 100000000:.2f} 亿", # 格式化为“亿”
        'MA120变异系数': round(ma120_cv, 4),
        '趋势斜率惩罚': round(trend_slope_pct, 4)
    }

def scan_etf_for_grid(min_years=3):
    """主程序：建议年限设为3年即可，太长会过滤掉很多优质新科技ETF"""
    etf_spot = get_etf_list_sina() 
    if etf_spot.empty:
        return
        
    total_etf = len(etf_spot)
    print(f"\n共发现 {total_etf} 只ETF。开始极速扫描，已开启【日均成交额>5000万】与【日振幅>1.5%】双重硬过滤...")
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=min_years * 365)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    results =[]
    count = 0
    valid_count = 0
    
    for index, row in etf_spot.iterrows():
        count += 1
        code = str(row['代码']).strip()
        name = str(row['名称']).strip()
        
        # 排除一眼假的品种
        if any(keyword in name for keyword in["货币", "债", "理财", "黄金", "添益", "快线"]):
            continue
            
        hist_df = get_etf_hist_sina_safe(code, start_str, end_str)
        
        # 数据长度够不够 (1年约240交易日)
        if hist_df is not None and len(hist_df) >= (min_years * 240) * 0.9:
            metrics = calculate_grid_metrics(hist_df)
            
            if metrics: # 如果metrics不为None，说明通过了流动性和振幅的魔鬼测试
                metrics['代码'] = code
                metrics['名称'] = name
                results.append(metrics)
                valid_count += 1
        
        if count % 50 == 0 or count == total_etf:
            print(f"进度：已处理 {count}/{total_etf}，当前幸存的网格圣体：{valid_count} 只...")
            
        time.sleep(0.01) # 微小延迟保护接口

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("\n扫描结束：没有找到符合所有苛刻条件的ETF。")
        return None
        
    # 排序
    res_df = res_df.sort_values('网格综合评分', ascending=False).reset_index(drop=True)
    
    # 调整列显示顺序
    cols =['代码', '名称', '网格综合评分', '平均日振幅(%)', '震荡纯度(0-1)', '近20日均成交额', 'MA120变异系数', '趋势斜率惩罚']
    final_df = res_df[cols]
    
    print("\n================== 🎯 扫描完成！通过【资金面+技术面】双重考验的最终 TOP 15 ==================")
    print(final_df.head(15).to_string(index=False))
    
    # 导出完整的清洗结果
    final_df.to_csv("全市场网格ETF终极选品表.csv", index=False, encoding="utf-8-sig")
    print("\n完整结果已保存至：全市场网格ETF终极选品表.csv")
    
    return final_df

if __name__ == "__main__":
    # 注意：我们将历史考察期设为 3 年。
    # 因为很多高弹性的硬科技、医药ETF是近三年上市的，5年/10年会把最好的品种过滤掉。
    scan_etf_for_grid(min_years=3)