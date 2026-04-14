"""
HOÀN THIỆN PIPELINE DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN - PHIÊN BẢN ĐÃ SỬA LỖI
Xử lý missing data, unseen categories, feature interaction, và so sánh mô hình numerical/text/image
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==================== PHẦN 1: XỬ LÝ MISSING + UNSEEN CATEGORIES ====================

def handle_missing_and_unseen(train_df, test_df, cat_cols, num_cols, group_col='district'):
    """
    Xử lý missing values và unseen categories cho train/test
    """
    print("Đang xử lý missing values và unseen categories...")
    
    # Copy để tránh ảnh hưởng data gốc
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # 1. Xử lý missing numeric theo nhóm
    for col in num_cols:
        if col in train_df.columns:
            # Theo nhóm district
            train_df[col] = train_df.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.median()) if x.notna().any() else x
            )
            test_df[col] = test_df.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.median()) if x.notna().any() else x
            )
            # Fallback to overall median
            if train_df[col].isna().any():
                train_df[col].fillna(train_df[col].median(), inplace=True)
            if test_df[col].isna().any():
                test_df[col].fillna(train_df[col].median(), inplace=True)
    
    # 2. Xử lý missing categorical
    for col in cat_cols:
        if col in train_df.columns:
            train_df[col].fillna('Unknown', inplace=True)
            test_df[col].fillna('Unknown', inplace=True)
    
    # 3. Xử lý unseen categories
    for col in cat_cols:
        if col in train_df.columns:
            # Gom nhóm hiếm trong train thành 'Other'
            freq = train_df[col].value_counts()
            rare_cats = freq[freq < 5].index
            train_df[col] = train_df[col].replace(rare_cats, 'Other')
            
            # Test: unseen = không có trong train
            known_cats = set(train_df[col].unique())
            test_df[col] = test_df[col].apply(
                lambda x: x if x in known_cats else 'Other'
            )
            
            # Label encode
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            # Xử lý test: map hoặc gán -1
            test_df[col] = test_df[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
            )
    
    print("  Hoàn thành xử lý missing và unseen!")
    return train_df, test_df


# ==================== PHẦN 2: FEATURE INTERACTION ====================

def add_feature_interaction(df, area_col='area', room_col='rooms', district_col='district'):
    """
    Tạo feature interaction: diện tích × số phòng × quận
    """
    print("\nĐang tạo feature interaction...")
    df = df.copy()
    
    # Interaction 1: area * rooms
    if area_col in df.columns and room_col in df.columns:
        df['area_rooms_interaction'] = df[area_col] * df[room_col]
        print(f"  Đã tạo feature: area_rooms_interaction")
    
    # Interaction 2: area * district (dùng target encoding đơn giản)
    if area_col in df.columns and district_col in df.columns:
        # Tính mean area theo district
        district_mean = df.groupby(district_col)[area_col].mean().to_dict()
        df['area_district_interaction'] = df[district_col].map(district_mean) * df[area_col]
        print(f"  Đã tạo feature: area_district_interaction")
    
    # Interaction 3: rooms * district
    if room_col in df.columns and district_col in df.columns:
        district_room_mean = df.groupby(district_col)[room_col].mean().to_dict()
        df['rooms_district_interaction'] = df[district_col].map(district_room_mean) * df[room_col]
        print(f"  Đã tạo feature: rooms_district_interaction")
    
    print("  Hoàn thành tạo feature interaction!")
    return df


def evaluate_interaction_improvement(X_train, y_train, X_test, y_test):
    """
    Đánh giá cải thiện khi thêm interaction features
    """
    print("\n" + "="*60)
    print("ĐÁNH GIÁ CẢI THIỆN FEATURE INTERACTION")
    print("="*60)
    
    # KIỂM TRA VÀ XỬ LÝ NaN TRONG DỮ LIỆU
    print("\nKiểm tra dữ liệu đầu vào...")
    print(f"  X_train NaN: {np.isnan(X_train).sum()}")
    print(f"  X_test NaN: {np.isnan(X_test).sum()}")
    
    # Xóa các hàng có NaN trong y
    valid_train = ~np.isnan(y_train)
    valid_test = ~np.isnan(y_test)
    
    X_train_clean = X_train[valid_train]
    y_train_clean = y_train[valid_train]
    X_test_clean = X_test[valid_test]
    y_test_clean = y_test[valid_test]
    
    # Xóa các hàng có NaN trong X
    valid_train_x = ~np.isnan(X_train_clean).any(axis=1)
    valid_test_x = ~np.isnan(X_test_clean).any(axis=1)
    
    X_train_final = X_train_clean[valid_train_x]
    y_train_final = y_train_clean[valid_train_x]
    X_test_final = X_test_clean[valid_test_x]
    y_test_final = y_test_clean[valid_test_x]
    
    print(f"  Sau khi làm sạch: Train={len(X_train_final)}, Test={len(X_test_final)}")
    
    # Mô hình không interaction
    print("\nĐang huấn luyện mô hình KHÔNG interaction...")
    model_base = Ridge(alpha=1.0)
    model_base.fit(X_train_final, y_train_final)
    y_pred_base = model_base.predict(X_test_final)
    r2_base = r2_score(y_test_final, y_pred_base)
    rmse_base = np.sqrt(mean_squared_error(y_test_final, y_pred_base))
    
    # Tạo interaction features
    print("Đang tạo interaction features...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_inter = poly.fit_transform(X_train_final)
    X_test_inter = poly.transform(X_test_final)
    
    # Mô hình có interaction
    print("Đang huấn luyện mô hình CÓ interaction...")
    model_inter = Ridge(alpha=1.0)
    model_inter.fit(X_train_inter, y_train_final)
    y_pred_inter = model_inter.predict(X_test_inter)
    r2_inter = r2_score(y_test_final, y_pred_inter)
    rmse_inter = np.sqrt(mean_squared_error(y_test_final, y_pred_inter))
    
    # Kết quả
    print(f"\n📊 KẾT QUẢ:")
    print(f"\nMô hình KHÔNG interaction:")
    print(f"  R² Score: {r2_base:.4f}")
    print(f"  RMSE: {rmse_base:.2f}")
    
    print(f"\nMô hình CÓ interaction (bậc 2):")
    print(f"  R² Score: {r2_inter:.4f}")
    print(f"  RMSE: {rmse_inter:.2f}")
    
    print(f"\n📈 CẢI THIỆN:")
    print(f"  R² tăng: {(r2_inter - r2_base)*100:+.2f}%")
    print(f"  RMSE giảm: {abs(rmse_base - rmse_inter):+.2f}")
    
    return {
        'r2_base': r2_base,
        'r2_inter': r2_inter,
        'rmse_base': rmse_base,
        'rmse_inter': rmse_inter,
        'improvement': r2_inter - r2_base
    }


# ==================== PHẦN 3: SO SÁNH MÔ HÌNH ====================

def compare_models(X_num_train, X_num_test, y_train, y_test, 
                   text_train=None, text_test=None,
                   img_features_train=None, img_features_test=None):
    """
    So sánh 3 loại mô hình:
    1. Chỉ numerical
    2. Numerical + Text
    3. Numerical + Text + Image
    """
    print("\n" + "="*60)
    print("SO SÁNH CÁC LOẠI MÔ HÌNH")
    print("="*60)
    
    # Làm sạch dữ liệu
    valid_train = ~(np.isnan(y_train) | np.isnan(X_num_train).any(axis=1))
    valid_test = ~(np.isnan(y_test) | np.isnan(X_num_test).any(axis=1))
    
    X_num_train_clean = X_num_train[valid_train]
    X_num_test_clean = X_num_test[valid_test]
    y_train_clean = y_train[valid_train]
    y_test_clean = y_test[valid_test]
    
    results = {}
    
    # 1. Mô hình chỉ numerical
    print("\n1. MÔ HÌNH CHỈ NUMERICAL FEATURES")
    model_num = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_num.fit(X_num_train_clean, y_train_clean)
    y_pred_num = model_num.predict(X_num_test_clean)
    
    r2_num = r2_score(y_test_clean, y_pred_num)
    rmse_num = np.sqrt(mean_squared_error(y_test_clean, y_pred_num))
    mae_num = mean_absolute_error(y_test_clean, y_pred_num)
    
    print(f"  R² Score: {r2_num:.4f}")
    print(f"  RMSE: {rmse_num:.2f}")
    print(f"  MAE: {mae_num:.2f}")
    
    results['numerical'] = {'r2': r2_num, 'rmse': rmse_num, 'mae': mae_num}
    
    # 2. Mô hình Numerical + Text
    if text_train is not None and text_test is not None:
        print("\n2. MÔ HÌNH NUMERICAL + TEXT FEATURES")
        
        # Lọc text theo cùng index
        text_train_clean = [text_train[i] for i in range(len(text_train)) if valid_train[i]]
        text_test_clean = [text_test[i] for i in range(len(text_test)) if valid_test[i]]
        
        # TF-IDF cho text
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        X_text_train = tfidf.fit_transform(text_train_clean).toarray()
        X_text_test = tfidf.transform(text_test_clean).toarray()
        
        # Kết hợp features
        X_num_text_train = np.hstack([X_num_train_clean, X_text_train])
        X_num_text_test = np.hstack([X_num_test_clean, X_text_test])
        
        model_text = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_text.fit(X_num_text_train, y_train_clean)
        y_pred_text = model_text.predict(X_num_text_test)
        
        r2_text = r2_score(y_test_clean, y_pred_text)
        rmse_text = np.sqrt(mean_squared_error(y_test_clean, y_pred_text))
        mae_text = mean_absolute_error(y_test_clean, y_pred_text)
        
        print(f"  R² Score: {r2_text:.4f}")
        print(f"  RMSE: {rmse_text:.2f}")
        print(f"  MAE: {mae_text:.2f}")
        
        results['numerical_text'] = {'r2': r2_text, 'rmse': rmse_text, 'mae': mae_text}
        
        # 3. Mô hình Numerical + Text + Image
        if img_features_train is not None and img_features_test is not None:
            print("\n3. MÔ HÌNH NUMERICAL + TEXT + IMAGE FEATURES")
            
            img_train_clean = img_features_train[valid_train]
            img_test_clean = img_features_test[valid_test]
            
            X_all_train = np.hstack([X_num_text_train, img_train_clean])
            X_all_test = np.hstack([X_num_text_test, img_test_clean])
            
            model_all = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model_all.fit(X_all_train, y_train_clean)
            y_pred_all = model_all.predict(X_all_test)
            
            r2_all = r2_score(y_test_clean, y_pred_all)
            rmse_all = np.sqrt(mean_squared_error(y_test_clean, y_pred_all))
            mae_all = mean_absolute_error(y_test_clean, y_pred_all)
            
            print(f"  R² Score: {r2_all:.4f}")
            print(f"  RMSE: {rmse_all:.2f}")
            print(f"  MAE: {mae_all:.2f}")
            
            results['numerical_text_image'] = {'r2': r2_all, 'rmse': rmse_all, 'mae': mae_all}
    
    # Tổng kết so sánh
    print("\n" + "="*60)
    print("TỔNG KẾT SO SÁNH")
    print("="*60)
    print(f"{'Model Type':<30} {'R²':<10} {'RMSE':<12} {'MAE':<12}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['r2']:<10.4f} {metrics['rmse']:<12.2f} {metrics['mae']:<12.2f}")
    
    return results


# ==================== PHẦN 4: TẠO DỮ LIỆU MẪU ====================

def create_sample_data(n_samples=1000):
    """
    Tạo dữ liệu mẫu cho demo
    """
    np.random.seed(42)
    
    data = {
        'area': np.random.uniform(30, 150, n_samples),
        'rooms': np.random.randint(1, 6, n_samples),
        'year_built': np.random.randint(1980, 2024, n_samples),
        'floor': np.random.randint(1, 15, n_samples),
        'district': np.random.choice(['Quận 1', 'Quận 2', 'Quận 3', 'Quận 4', 'Quận 5', 'Quận 7', 'Quận 9', 'Quận Tân Bình', 'Quận Bình Thạnh'], n_samples),
        'description': [
            f"Nhà đẹp tại {np.random.choice(['trung tâm', 'ngoại ô', 'gần chợ', 'gần trường', 'mặt tiền'])}" 
            for _ in range(n_samples)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Thêm missing values (khoảng 5%)
    for col in ['area', 'rooms', 'district', 'description']:
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    
    # Tạo target (giá) dựa trên features
    df['price'] = (df['area'].fillna(80) * 20 + 
                   df['rooms'].fillna(3) * 500 + 
                   (df['year_built'].fillna(2000) - 1980) * 10 + 
                   np.random.normal(0, 100, n_samples))
    
    return df


# ==================== PHẦN 5: HÀM MAIN CHẠY TOÀN BỘ PIPELINE ====================

def main():
    """
    Chạy toàn bộ pipeline hoàn thiện
    """
    print("\n" + "="*60)
    print("BẮT ĐẦU PIPELINE HOÀN THIỆN")
    print("="*60)
    
    # 1. Tạo dữ liệu mẫu
    print("\n1. TẠO DỮ LIỆU MẪU...")
    df_full = create_sample_data(1000)
    print(f"   Kích thước dataset: {df_full.shape}")
    print(f"   Missing values:\n{df_full.isnull().sum()}")
    
    # 2. Chia train/test
    train_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42)
    print(f"\n   Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # 3. Xác định columns
    num_cols = ['area', 'rooms', 'year_built', 'floor']
    cat_cols = ['district']
    target_col = 'price'
    text_col = 'description'
    
    # 4. Xử lý missing và unseen categories
    train_df, test_df = handle_missing_and_unseen(
        train_df, test_df, cat_cols, num_cols, group_col='district'
    )
    
    # 5. Feature interaction
    train_df = add_feature_interaction(train_df, 'area', 'rooms', 'district')
    test_df = add_feature_interaction(test_df, 'area', 'rooms', 'district')
    
    # Cập nhật numerical columns sau khi thêm interaction
    num_cols_extended = num_cols + ['area_rooms_interaction', 'area_district_interaction', 'rooms_district_interaction']
    
    # 6. Chuẩn bị dữ liệu cho models - ĐẢM BẢO KHÔNG CÒN NaN
    print("\nChuẩn bị dữ liệu cho models...")
    
    X_num_train = train_df[num_cols_extended].values.astype(float)
    X_num_test = test_df[num_cols_extended].values.astype(float)
    y_train = train_df[target_col].values.astype(float)
    y_test = test_df[target_col].values.astype(float)
    
    # Kiểm tra và xử lý NaN lần cuối
    print(f"  NaN trong X_num_train: {np.isnan(X_num_train).sum()}")
    print(f"  NaN trong X_num_test: {np.isnan(X_num_test).sum()}")
    
    # Thay thế NaN còn sót bằng 0
    X_num_train = np.nan_to_num(X_num_train, nan=0.0)
    X_num_test = np.nan_to_num(X_num_test, nan=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0)
    
    print(f"  Sau xử lý NaN: X_num_train shape={X_num_train.shape}, X_num_test shape={X_num_test.shape}")
    
    # 7. Đánh giá feature interaction (chỉ dùng 3 features gốc)
    X_train_3features = X_num_train[:, :3]  # area, rooms, year_built
    X_test_3features = X_num_test[:, :3]
    
    interaction_results = evaluate_interaction_improvement(
        X_train_3features, y_train, X_test_3features, y_test
    )
    
    # 8. So sánh các mô hình
    text_train = train_df[text_col].fillna('').values
    text_test = test_df[text_col].fillna('').values
    
    # Demo với image features (tạo random để demo)
    print("\n" + "="*60)
    print("CHUẨN BỊ IMAGE FEATURES (DEMO)")
    print("="*60)
    print("Lưu ý: Trong thực tế, bạn cần extract features từ ảnh thật bằng CNN")
    img_features_train = np.random.randn(len(train_df), 512).astype(float)
    img_features_test = np.random.randn(len(test_df), 512).astype(float)
    print(f"  Image features shape: {img_features_train.shape}")
    
    # So sánh models
    compare_results = compare_models(
        X_num_train, X_num_test, y_train, y_test,
        text_train, text_test,
        img_features_train, img_features_test
    )
    
    print("\n" + "="*60)
    print("✅ PIPELINE HOÀN THIỆN ĐÃ CHẠY THÀNH CÔNG!")
    print("="*60)
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'interaction_results': interaction_results,
        'compare_results': compare_results
    }


# ==================== CHẠY CHƯƠNG TRÌNH ====================

if __name__ == "__main__":
    try:
        results = main()
        
        # In thêm thông tin chi tiết
        print("\n" + "="*60)
        print("THÔNG TIN CHI TIẾT KẾT QUẢ")
        print("="*60)
        print("\n📊 Feature Interaction Results:")
        for key, value in results['interaction_results'].items():
            print(f"   {key}: {value:.4f}")
        
        print("\n📊 Model Comparison Results:")
        for model_name, metrics in results['compare_results'].items():
            print(f"\n   {model_name}:")
            for metric, value in metrics.items():
                print(f"      {metric}: {value:.4f}")
                
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()