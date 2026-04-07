"""
PROPTECH DATA PIPELINE - HOÀN CHỈNH
Xử lý dữ liệu bất động sản từ nhiều nguồn, làm sạch, feature engineering, phát hiện duplicate,
và tạo insights cho dự đoán giá nhà.
"""

import pandas as pd
import numpy as np
import json
import requests
import os
import warnings
from datetime import datetime
from difflib import SequenceMatcher
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================
# PHẦN 1: ĐỌC DỮ LIỆU TỪ NHIỀU NGUỒN
# ============================================================

class DataIngestor:
    """Đọc dữ liệu từ CSV, JSON, API, Excel"""
    
    @staticmethod
    def load_csv(file_path):
        print(f"  📄 Đọc CSV: {file_path}")
        return pd.read_csv(file_path)
    
    @staticmethod
    def load_json(file_path):
        print(f"  📄 Đọc JSON: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
    @staticmethod
    def load_excel(file_path):
        print(f"  📄 Đọc Excel: {file_path}")
        return pd.read_excel(file_path)
    
    @staticmethod
    def load_api(url, params=None, headers=None):
        print(f"  🌐 Gọi API: {url}")
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                return pd.DataFrame([data])
        except Exception as e:
            print(f"  ⚠️ Lỗi API: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def load_from_multiple_sources(sources):
        """
        sources: list of dict, mỗi dict có dạng:
        {'type': 'csv', 'path': '...'} hoặc {'type': 'api', 'url': '...'}
        """
        dfs = []
        for source in sources:
            if source['type'] == 'csv':
                df = DataIngestor.load_csv(source['path'])
            elif source['type'] == 'json':
                df = DataIngestor.load_json(source['path'])
            elif source['type'] == 'excel':
                df = DataIngestor.load_excel(source['path'])
            elif source['type'] == 'api':
                df = DataIngestor.load_api(source['url'])
            else:
                continue
            if not df.empty:
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()


# ============================================================
# PHẦN 2: LÀM SẠCH DỮ LIỆU CƠ BẢN
# ============================================================

class DataCleaner:
    """Làm sạch và chuẩn hóa dữ liệu"""
    
    @staticmethod
    def standardize_columns(df):
        """Chuẩn hóa tên cột: lowercase, bỏ khoảng trắng, thay khoảng trắng bằng _"""
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df
    
    @staticmethod
    def remove_duplicates(df, subset=None):
        """Xóa bản ghi trùng lặp hoàn hảo"""
        before = len(df)
        df = df.drop_duplicates(subset=subset)
        after = len(df)
        print(f"  🗑️ Đã xóa {before - after} dòng trùng lặp")
        return df
    
    @staticmethod
    def handle_missing_values(df, numeric_strategy='median', categorical_strategy='mode'):
        """Xử lý missing values"""
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            if df[col].dtype in ['int64', 'float64']:
                if numeric_strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif numeric_strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif numeric_strategy == 'zero':
                    df[col].fillna(0, inplace=True)
            else:
                if categorical_strategy == 'mode':
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                    df[col].fillna(mode_val, inplace=True)
                elif categorical_strategy == 'constant':
                    df[col].fillna('unknown', inplace=True)
            print(f"  📍 Đã xử lý {missing_count} missing ở cột '{col}'")
        return df
    
    @staticmethod
    def standardize_categorical(df, categorical_cols):
        """Chuẩn hóa text dạng categorical: lowercase, strip"""
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col] = df[col].replace('nan', 'unknown')
        return df
    
    @staticmethod
    def filter_valid_range(df, column, min_val=None, max_val=None):
        """Lọc giá trị hợp lệ trong khoảng"""
        before = len(df)
        if min_val is not None:
            df = df[df[column] >= min_val]
        if max_val is not None:
            df = df[df[column] <= max_val]
        after = len(df)
        print(f"  🎯 Cột '{column}': giữ lại {after}/{before} dòng")
        return df
    
    @staticmethod
    def clean_all(df, required_columns=None):
        """Pipeline làm sạch tổng hợp"""
        print("\n🧹 BẮT ĐẦU LÀM SẠCH DỮ LIỆU...")
        
        df = DataCleaner.standardize_columns(df)
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"  ⚠️ Thiếu cột bắt buộc: {missing_cols}")
        
        df = DataCleaner.remove_duplicates(df)
        df = DataCleaner.handle_missing_values(df)
        
        return df


# ============================================================
# PHẦN 3: PHÁT HIỆN VÀ XỬ LÝ OUTLIER
# ============================================================

class OutlierHandler:
    """Phát hiện và xử lý outlier bằng IQR và Z-score"""
    
    @staticmethod
    def detect_outliers_iqr(df, column):
        """Phát hiện outlier bằng IQR"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        return outliers, lower_bound, upper_bound
    
    @staticmethod
    def detect_outliers_zscore(df, column, threshold=3):
        """Phát hiện outlier bằng Z-score"""
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = np.abs(stats.zscore(df[column])) > threshold
        return outliers
    
    @staticmethod
    def handle_outliers(df, column, strategy='capping', method='iqr'):
        """
        Xử lý outlier
        strategy: 'remove', 'capping', 'log_transform'
        method: 'iqr', 'zscore'
        """
        before_count = len(df)
        
        if method == 'iqr':
            outliers, lower, upper = OutlierHandler.detect_outliers_iqr(df, column)
        else:
            outliers = OutlierHandler.detect_outliers_zscore(df, column)
            lower, upper = df[column].quantile(0.01), df[column].quantile(0.99)
        
        outlier_count = outliers.sum()
        
        if strategy == 'remove':
            df = df[~outliers]
            print(f"  🗑️ Đã xóa {outlier_count} outlier ở cột '{column}'")
            
        elif strategy == 'capping':
            df[column] = df[column].clip(lower, upper)
            print(f"  📦 Đã capping {outlier_count} outlier ở cột '{column}'")
            
        elif strategy == 'log_transform':
            df[column] = np.log1p(df[column])
            print(f"  📊 Đã log-transform cột '{column}' để giảm skew")
        
        after_count = len(df)
        return df
    
    @staticmethod
    def handle_all_outliers(df, numerical_cols, strategy='capping'):
        """Xử lý outlier cho tất cả cột số"""
        print("\n📊 PHÁT HIỆN VÀ XỬ LÝ OUTLIER...")
        for col in numerical_cols:
            if col in df.columns:
                df = OutlierHandler.handle_outliers(df, col, strategy)
        return df


# ============================================================
# PHẦN 4: FEATURE ENGINEERING
# ============================================================

class FeatureEngineer:
    """Chuẩn hóa số, biến đổi categorical, text processing"""
    
    @staticmethod
    def scale_numerical(df, columns, method='minmax'):
        """
        Chuẩn hóa dữ liệu số
        method: 'minmax' hoặc 'zscore'
        """
        print(f"\n📏 CHUẨN HÓA DỮ LIỆU SỐ ({method})...")
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        available_cols = [col for col in columns if col in df.columns]
        if available_cols:
            df[available_cols] = scaler.fit_transform(df[available_cols])
            print(f"  ✅ Đã chuẩn hóa: {available_cols}")
        
        return df, scaler
    
    @staticmethod
    def encode_categorical(df, columns, method='onehot'):
        """
        Biến đổi categorical
        method: 'onehot' hoặc 'label'
        """
        print(f"\n🏷️ ENCODING CATEGORICAL ({method})...")
        
        if method == 'onehot':
            available_cols = [col for col in columns if col in df.columns]
            if available_cols:
                dummies = pd.get_dummies(df[available_cols], prefix=available_cols)
                df = pd.concat([df.drop(columns=available_cols), dummies], axis=1)
                print(f"  ✅ One-hot encoding cho: {available_cols}")
        else:  # label encoding
            for col in columns:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    print(f"  ✅ Label encoding cho: {col}")
        
        return df
    
    @staticmethod
    def vectorize_text(df, text_column='description', max_features=100, ngram_range=(1, 2)):
        """
        Biến đổi text thành vector TF-IDF
        """
        print(f"\n📝 TEXT VECTORIZATION (TF-IDF)...")
        
        if text_column not in df.columns:
            print(f"  ⚠️ Không tìm thấy cột '{text_column}', bỏ qua TF-IDF")
            return df, None
        
        tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=ngram_range,
            lowercase=True
        )
        
        text_features = tfidf.fit_transform(df[text_column].fillna(''))
        text_df = pd.DataFrame(
            text_features.toarray(),
            columns=[f'text_{i}' for i in range(max_features)]
        )
        
        df = pd.concat([df.reset_index(drop=True), text_df], axis=1)
        print(f"  ✅ Đã tạo {max_features} đặc trưng TF-IDF từ text")
        
        return df, tfidf
    
    @staticmethod
    def create_derived_features(df):
        """Tạo các đặc trưng suy dẫn từ dữ liệu có sẵn"""
        print("\n🔧 TẠO ĐẶC TRƯNG SUY DẪN...")
        
        # Giá trên m2
        if 'price' in df.columns and 'area' in df.columns:
            df['price_per_sqm'] = df['price'] / df['area']
            print("  ✅ Đã tạo 'price_per_sqm'")
        
        # Tổng số phòng (nếu có bedroom + livingroom)
        if 'bedrooms' in df.columns and 'livingrooms' in df.columns:
            df['total_rooms'] = df['bedrooms'] + df['livingrooms']
            print("  ✅ Đã tạo 'total_rooms'")
        
        # Bình phương diện tích (cho mô hình phi tuyến)
        if 'area' in df.columns:
            df['area_squared'] = df['area'] ** 2
            print("  ✅ Đã tạo 'area_squared'")
        
        # Log của giá (nếu skew cao)
        if 'price' in df.columns:
            df['log_price'] = np.log1p(df['price'])
            print("  ✅ Đã tạo 'log_price'")
        
        return df


# ============================================================
# PHẦN 5: PHÁT HIỆN DUPLICATE DỰA TRÊN TEXT SIMILARITY
# ============================================================

class DuplicateDetector:
    """Phát hiện và merge các bản ghi trùng lặp dựa trên text similarity"""
    
    @staticmethod
    def text_similarity(text1, text2):
        """Tính độ tương đồng giữa hai chuỗi text"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    @staticmethod
    def find_duplicate_pairs(df, text_column='description', threshold=0.85, max_pairs=1000):
        """
        Tìm các cặp bản ghi trùng lặp dựa trên text
        """
        print(f"\n🔍 TÌM DUPLICATE DỰA TRÊN TEXT SIMILARITY (threshold={threshold})...")
        
        if text_column not in df.columns:
            print(f"  ⚠️ Không tìm thấy cột '{text_column}', bỏ qua")
            return []
        
        duplicates = []
        n = len(df)
        
        # Chỉ xét sample nếu dữ liệu quá lớn
        if n > 500:
            print(f"  ⚠️ Dữ liệu lớn ({n} dòng), chỉ xét 500 dòng đầu")
            n = min(500, n)
        
        for i in range(n):
            text_i = df.iloc[i][text_column]
            if pd.isna(text_i):
                continue
            for j in range(i + 1, n):
                text_j = df.iloc[j][text_column]
                if pd.isna(text_j):
                    continue
                sim = DuplicateDetector.text_similarity(text_i, text_j)
                if sim > threshold:
                    duplicates.append((i, j, sim))
                    if len(duplicates) >= max_pairs:
                        break
            if len(duplicates) >= max_pairs:
                break
        
        print(f"  ✅ Tìm thấy {len(duplicates)} cặp duplicate tiềm năng")
        return duplicates
    
    @staticmethod
    def merge_duplicate_records(df, duplicate_pairs, priority_column='price'):
        """
        Merge các bản ghi trùng lặp
        priority_column: cột dùng để quyết định giữ record nào (giữ record có giá trị lớn hơn)
        """
        if not duplicate_pairs:
            return df
        
        print(f"\n🔄 MERGE {len(duplicate_pairs)} CẶP DUPLICATE...")
        
        to_drop = set()
        
        for i, j, sim in duplicate_pairs:
            # Giữ record có giá trị priority_column cao hơn
            if priority_column in df.columns:
                if df.iloc[i][priority_column] < df.iloc[j][priority_column]:
                    to_drop.add(i)
                else:
                    to_drop.add(j)
            else:
                # Nếu không có priority_column, giữ record có description dài hơn
                desc_i = str(df.iloc[i].get('description', ''))
                desc_j = str(df.iloc[j].get('description', ''))
                if len(desc_i) < len(desc_j):
                    to_drop.add(i)
                else:
                    to_drop.add(j)
        
        df_merged = df.drop(index=to_drop).reset_index(drop=True)
        print(f"  ✅ Đã xóa {len(to_drop)} bản ghi trùng lặp, còn {len(df_merged)} dòng")
        
        return df_merged
    
    @staticmethod
    def auto_merge_duplicates(df, text_column='description', threshold=0.85):
        """Tự động phát hiện và merge duplicate"""
        duplicate_pairs = DuplicateDetector.find_duplicate_pairs(df, text_column, threshold)
        if duplicate_pairs:
            df = DuplicateDetector.merge_duplicate_records(df, duplicate_pairs)
        else:
            print("  ℹ️ Không tìm thấy duplicate nào")
        return df


# ============================================================
# PHẦN 6: MÔ HÌNH DỰ ĐOÁN GIÁ NHÀ
# ============================================================

class PricePredictor:
    """Xây dựng mô hình dự đoán giá nhà"""
    
    @staticmethod
    def prepare_features(df, target_column='price', exclude_columns=None):
        """Chuẩn bị features cho mô hình"""
        if exclude_columns is None:
            exclude_columns = []
        
        exclude_columns = exclude_columns + [target_column, 'log_price']
        
        # Lấy tất cả cột numeric
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_columns]
        
        X = df[feature_cols].fillna(0)
        y = df[target_column]
        
        print(f"\n🎯 Chuẩn bị features: {len(feature_cols)} đặc trưng")
        return X, y
    
    @staticmethod
    def train_model(X, y, test_size=0.2, random_state=42):
        """Huấn luyện mô hình Random Forest"""
        print("\n🤖 HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN GIÁ...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Đánh giá
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        print(f"  📊 R² Score: {metrics['r2']:.4f}")
        print(f"  📊 MAE: {metrics['mae']:.2f}")
        print(f"  📊 RMSE: {metrics['rmse']:.2f}")
        
        return model, metrics
    
    @staticmethod
    def get_feature_importance(model, feature_names):
        """Lấy feature importance"""
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📈 TOP 10 FEATURE QUAN TRỌNG NHẤT:")
        for i, row in importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        return importance


# ============================================================
# PHẦN 7: INSIGHT NGHIỆP VỤ
# ============================================================

class BusinessInsights:
    """Tạo các chỉ số và insight cho PropTech"""
    
    @staticmethod
    def generate_insights(df):
        """Tạo các insight nghiệp vụ"""
        print("\n💡 TẠO INSIGHT NGHIỆP VỤ...")
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': df.shape,
            'basic_stats': {}
        }
        
        # Thống kê cơ bản
        if 'price' in df.columns:
            insights['basic_stats']['price'] = {
                'mean': df['price'].mean(),
                'median': df['price'].median(),
                'min': df['price'].min(),
                'max': df['price'].max(),
                'std': df['price'].std()
            }
        
        if 'area' in df.columns:
            insights['basic_stats']['area'] = {
                'mean': df['area'].mean(),
                'median': df['area'].median(),
                'min': df['area'].min(),
                'max': df['area'].max()
            }
        
        # Giá trên m2
        if 'price' in df.columns and 'area' in df.columns:
            price_per_sqm = df['price'] / df['area']
            insights['price_per_sqm'] = {
                'mean': price_per_sqm.mean(),
                'median': price_per_sqm.median(),
                'min': price_per_sqm.min(),
                'max': price_per_sqm.max()
            }
            print(f"  💰 Giá trung bình theo m²: {price_per_sqm.mean():,.0f}")
        
        # Phân phối theo số phòng
        if 'rooms' in df.columns:
            insights['rooms_distribution'] = df['rooms'].value_counts().sort_index().to_dict()
            print(f"  🏠 Phân phối số phòng: {insights['rooms_distribution']}")
        
        # Giá theo tình trạng
        if 'status' in df.columns and 'price' in df.columns:
            insights['price_by_status'] = df.groupby('status')['price'].mean().sort_values(ascending=False).to_dict()
            print(f"  📊 Giá theo tình trạng: {insights['price_by_status']}")
        
        # Top khu vực
        if 'location' in df.columns:
            insights['top_locations'] = df['location'].value_counts().head(5).to_dict()
            print(f"  📍 Top khu vực: {insights['top_locations']}")
        
        # Missing data report
        missing_report = df.isnull().sum()
        insights['missing_data'] = missing_report[missing_report > 0].to_dict()
        
        return insights
    
    @staticmethod
    def save_insights(insights, output_path='data/processed/insights.json'):
        """Lưu insights ra file JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Đã lưu insights tại: {output_path}")


# ============================================================
# PHẦN 8: PIPELINE CHÍNH
# ============================================================

class PropTechPipeline:
    """Pipeline xử lý dữ liệu hoàn chỉnh cho PropTech"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.df = None
        self.model = None
        self.insights = None
    
    def run(self, sources, target_column='price'):
        """
        Chạy toàn bộ pipeline
        
        sources: list of dict, ví dụ:
        [
            {'type': 'csv', 'path': 'data/houses.csv'},
            {'type': 'json', 'path': 'data/properties.json'},
            {'type': 'api', 'url': 'https://api.example.com/listings'}
        ]
        """
        print("=" * 60)
        print("🏢 PROPTECH DATA PIPELINE - BẮT ĐẦU")
        print("=" * 60)
        
        # 1. Đọc dữ liệu
        print("\n📂 BƯỚC 1: ĐỌC DỮ LIỆU TỪ CÁC NGUỒN")
        self.df = DataIngestor.load_from_multiple_sources(sources)
        print(f"  ✅ Đã đọc {len(self.df)} bản ghi")
        
        if self.df.empty:
            print("  ❌ Không có dữ liệu, pipeline dừng lại")
            return None
        
        # 2. Làm sạch cơ bản
        required_cols = ['price', 'area']
        self.df = DataCleaner.clean_all(self.df, required_cols)
        
        # 3. Lọc giá trị hợp lý
        if 'price' in self.df.columns:
            self.df = DataCleaner.filter_valid_range(self.df, 'price', min_val=10000, max_val=1e10)
        if 'area' in self.df.columns:
            self.df = DataCleaner.filter_valid_range(self.df, 'area', min_val=5, max_val=10000)
        
        # 4. Xử lý outlier
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.df = OutlierHandler.handle_all_outliers(self.df, numerical_cols, strategy='capping')
        
        # 5. Phát hiện duplicate bằng text
        self.df = DuplicateDetector.auto_merge_duplicates(self.df, text_column='description', threshold=0.85)
        
        # 6. Feature Engineering
        # 6a. Tạo đặc trưng suy dẫn
        self.df = FeatureEngineer.create_derived_features(self.df)
        
        # 6b. Chuẩn hóa số
        scale_cols = ['price', 'area'] if 'area' in self.df.columns else ['price']
        scale_cols = [c for c in scale_cols if c in self.df.columns]
        self.df, _ = FeatureEngineer.scale_numerical(self.df, scale_cols, method='minmax')
        
        # 6c. Encoding categorical
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        # Loại bỏ text column khỏi encoding (sẽ xử lý riêng bằng TF-IDF)
        if 'description' in categorical_cols:
            categorical_cols.remove('description')
        if categorical_cols:
            self.df = FeatureEngineer.encode_categorical(self.df, categorical_cols, method='onehot')
        
        # 6d. Text vectorization
        if 'description' in self.df.columns:
            self.df, _ = FeatureEngineer.vectorize_text(self.df, text_column='description', max_features=50)
        
        # 7. Tạo insights
        self.insights = BusinessInsights.generate_insights(self.df)
        BusinessInsights.save_insights(self.insights)
        
        # 8. Huấn luyện mô hình dự đoán giá
        if target_column in self.df.columns and len(self.df) > 100:
            X, y = PricePredictor.prepare_features(self.df, target_column)
            if len(X.columns) > 0:
                self.model, metrics = PricePredictor.train_model(X, y)
                self.insights['model_metrics'] = metrics
                BusinessInsights.save_insights(self.insights)
        
        # 9. Lưu dữ liệu đã xử lý
        output_path = 'data/processed/clean_data.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"\n💾 Đã lưu dữ liệu đã xử lý tại: {output_path}")
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE HOÀN THÀNH!")
        print("=" * 60)
        
        return self.df
    
    def predict_price(self, sample_data):
        """Dự đoán giá cho một bất động sản mới"""
        if self.model is None:
            print("⚠️ Chưa có mô hình, cần chạy pipeline trước")
            return None
        # TODO: Chuẩn hóa sample_data cùng format với training
        return self.model.predict([sample_data])[0]


# ============================================================
# PHẦN 9: TẠO DỮ LIỆU MẪU ĐỂ TEST
# ============================================================

def create_sample_data():
    """Tạo dữ liệu mẫu để test pipeline"""
    os.makedirs('data/raw', exist_ok=True)
    
    # Sample CSV data
    csv_data = pd.DataFrame({
        'price': [250000, 320000, 180000, 450000, 295000, 5000000, 15000],
        'area': [65, 85, 50, 120, 78, 500, 20],
        'rooms': [2, 3, 2, 4, 3, 6, 1],
        'bedrooms': [2, 3, 2, 4, 3, 5, 1],
        'livingrooms': [1, 1, 1, 1, 1, 2, 1],
        'status': ['good', 'excellent', 'needs_repair', 'excellent', 'good', 'new', 'needs_repair'],
        'location': ['District 1', 'District 2', 'District 7', 'District 1', 'District 2', 'District 1', 'District 9'],
        'description': [
            'Nice apartment in city center, near shopping mall',
            'Luxury house with garden and swimming pool',
            'Small apartment, need renovation, cheap price',
            'Modern villa, great view, fully furnished',
            'Comfortable house, quiet neighborhood',
            'Mansion with big garden, swimming pool, garage',
            'Old apartment, need repair, low price'
        ]
    })
    csv_data.to_csv('data/raw/houses.csv', index=False)
    
    # Sample JSON data
    json_data = [
        {'price': 380000, 'area': 95, 'rooms': 3, 'bedrooms': 3, 'livingrooms': 1,
         'status': 'excellent', 'location': 'District 3', 
         'description': 'New building, high quality interior'},
        {'price': 210000, 'area': 70, 'rooms': 2, 'bedrooms': 2, 'livingrooms': 1,
         'status': 'good', 'location': 'District 4',
         'description': 'Nice view, near river'}
    ]
    with open('data/raw/properties.json', 'w') as f:
        json.dump(json_data, f)
    
    print("✅ Đã tạo dữ liệu mẫu tại folder 'data/raw/'")
    return csv_data


# ============================================================
# PHẦN 10: MAIN - CHẠY PIPELINE
# ============================================================

def main():
    """Hàm chính để chạy pipeline"""
    
    # Tạo dữ liệu mẫu nếu chưa có
    if not os.path.exists('data/raw/houses.csv'):
        print("📦 Tạo dữ liệu mẫu để test...")
        create_sample_data()
    
    # Cấu hình các nguồn dữ liệu
    sources = [
        {'type': 'csv', 'path': 'data/raw/houses.csv'},
        {'type': 'json', 'path': 'data/raw/properties.json'},
        # {'type': 'api', 'url': 'https://api.example.com/listings'}  # Uncomment nếu có API thật
    ]
    
    # Khởi tạo và chạy pipeline
    pipeline = PropTechPipeline()
    df_clean = pipeline.run(sources, target_column='price')
    
    if df_clean is not None:
        print(f"\n📊 KẾT QUẢ CUỐI CÙNG: {df_clean.shape[0]} dòng, {df_clean.shape[1]} cột")
        print("\n📋 5 dòng đầu tiên của dữ liệu đã xử lý:")
        print(df_clean.head())
        
        # Hiển thị insights
        if pipeline.insights:
            print("\n📈 TỔNG QUAN INSIGHTS:")
            if 'basic_stats' in pipeline.insights:
                print(f"  - Giá trung bình: {pipeline.insights['basic_stats'].get('price', {}).get('mean', 'N/A'):,.0f}")
            if 'price_per_sqm' in pipeline.insights:
                print(f"  - Giá/m² trung bình: {pipeline.insights['price_per_sqm']['mean']:,.0f}")
            if 'model_metrics' in pipeline.insights:
                print(f"  - Mô hình R²: {pipeline.insights['model_metrics']['r2']:.4f}")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()