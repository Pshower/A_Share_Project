# tests/test_akshare_connection.py
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径，确保可以导入 src 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import akshare as ak
import sys
import time

def test_spot_em():
    """测试东方财富实时行情接口"""
    print("正在测试 akshare.stock_zh_a_spot_em() ...")
    try:
        start = time.time()
        df = ak.stock_zh_a_spot_em()
        elapsed = time.time() - start
        if df is not None and not df.empty:
            print(f"✅ 成功获取数据，共 {len(df)} 行，耗时 {elapsed:.2f} 秒")
            print("前 5 行数据预览：")
            print(df.head())
            return True
        else:
            print("❌ 获取数据为空，可能接口返回异常")
            return False
    except Exception as e:
        print(f"❌ 请求失败：{type(e).__name__}: {e}")
        return False
    
def test_stock_zh_a_hist():
    """测试东方财富历史行情接口"""
    print("正在测试 akshare.stock_zh_a_hist() ...")
    try:
        start = time.time()
        df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20240528', adjust="")
        elapsed = time.time() - start
        if df is not None and not df.empty:
            print(f"✅ 成功获取数据，共 {len(df)} 行，耗时 {elapsed:.2f} 秒")
            print("前 5 行数据预览：")
            print(df.head())
            return True
        else:
            print("❌ 获取数据为空，可能接口返回异常")
            return False
    except Exception as e:
        print(f"❌ 请求失败：{type(e).__name__}: {e}")
        return False

def test_stock_board_industry_name_em():
    """测试东方财富行业板块接口"""
    print("正在测试 akshare.stock_board_industry_name_em() ...")
    try:
        start = time.time()
        df = ak.stock_board_industry_name_em()
        elapsed = time.time() - start
        if df is not None and not df.empty:
            print(f"✅ 成功获取数据，共 {len(df)} 行，耗时 {elapsed:.2f} 秒")
            print("前 5 行数据预览：")
            print(df.head())
            return True
        else:
            print("❌ 获取数据为空，可能接口返回异常")
            return False
    except Exception as e:
        print(f"❌ 请求失败：{type(e).__name__}: {e}")
        return False
    
def test_stock_info_a_code_name():
    """测试东方财富股票列表接口"""
    print("正在测试 akshare.stock_info_a_code_name() ...")
    try:
        start = time.time()
        df = ak.stock_info_a_code_name()
        elapsed = time.time() - start
        if df is not None and not df.empty:
            print(f"✅ 成功获取数据，共 {len(df)} 行，耗时 {elapsed:.2f} 秒")
            print("前 5 行数据预览：")
            print(df.head())
            return True
        else:
            print("❌ 获取数据为空，可能接口返回异常")
            return False
    except Exception as e:
        print(f"❌ 请求失败：{type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("AKShare 连接测试")
    print("=" * 40)
    print("测试 1: 东方财富实时行情接口")
    print("=" * 40)
    success = test_spot_em()
    if not success:
        print("\n可能的原因及解决方案：")
        print("1. 网络连接问题：请检查是否能访问 https://quote.eastmoney.com")
        print("2. 服务器限制：请稍后再试，或更换网络环境（如手机热点）")
        print("3. IP 被封：等待几分钟后重试，或使用代理")
        print("4. 接口变动：请更新 akshare 到最新版：pip install --upgrade akshare")
        print("\n备选方案：尝试使用新浪接口 ak.stock_zh_a_spot()（但数据格式略有不同）")
    else:
        print("\n接口正常，可以继续使用。")

    print("\n" + "=" * 40)
    print("测试 2: 东方财富历史行情接口")
    print("=" * 40)
    success = test_stock_zh_a_hist()
    if not success:
        print("\n可能的原因及解决方案：")
        print("1. 网络连接问题：请检查是否能访问 https://quote.eastmoney.com")
        print("2. 服务器限制：请稍后再试，或更换网络环境（如手机热点）")
        print("3. IP 被封：等待几分钟后重试，或使用代理")
        print("4. 接口变动：请更新 akshare 到最新版：pip install --upgrade akshare")
        print("\n备选方案：尝试使用新浪接口 ak.stock_zh_a_hist()（但数据格式略有不同）")
    else:
        print("\n接口正常，可以继续使用。")

    print("\n" + "=" * 40)
    print("测试 3: 东方财富行业板块接口")
    print("=" * 40)
    success = test_stock_board_industry_name_em()
    if not success:
        print("\n可能的原因及解决方案：")
        print("1. 网络连接问题：请检查是否能访问 https://quote.eastmoney.com")
        print("2. 服务器限制：请稍后再试，或更换网络环境（如手机热点）")
        print("3. IP 被封：等待几分钟后重试，或使用代理")
        print("4. 接口变动：请更新 akshare 到最新版：pip install --upgrade akshare")
        print("\n备选方案：尝试使用新浪接口 ak.stock_board_industry_name_em()（但数据格式略有不同）")
    else:
        print("\n接口正常，可以继续使用。")

    print("\n" + "=" * 40)
    print("测试 4: 东方财富股票列表接口")
    print("=" * 40)
    success = test_stock_info_a_code_name()
    if not success:
        print("\n可能的原因及解决方案：")
        print("1. 网络连接问题：请检查是否能访问 https://quote.eastmoney.com")
        print("2. 服务器限制：请稍后再试，或更换网络环境（如手机热点）")
        print("3. IP 被封：等待几分钟后重试，或使用代理")
        print("4. 接口变动：请更新 akshare 到最新版：pip install --upgrade akshare")
        print("\n备选方案：尝试使用新浪接口 ak.stock_info_a_code_name()（但数据格式略有不同）")
    else:
        print("\n接口正常，可以继续使用。")

    sys.exit(0 if success else 1)