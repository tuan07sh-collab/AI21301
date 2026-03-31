import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# PHẦN 2 – Tạo dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Score": [2, 4, 5, 6, 7, 8, 8.5, 9]
}
df = pd.DataFrame(data)

# PHẦN 3 – Tách Input (X) và Output (y)
X = df[["Hours"]] 
y = df["Score"]   

# PHẦN 4 & 5 – Tạo và Huấn luyện Model
model = LinearRegression()
model.fit(X, y) 

# PHẦN 6 – Dự đoán 1 giá trị mới
new_hours = [[6]]
predicted_score = model.predict(new_hours)
print(f"Dự đoán học 6 giờ: {predicted_score[0]:.2f} điểm")

# PHẦN 7 – Dự đoán nhiều giá trị
new_data = [[4], [6], [9]]
predictions = model.predict(new_data)
print("Dự đoán cho 4, 6, 9 giờ:", predictions)

# PHẦN 8 – Vẽ biểu đồ
plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, model.predict(X), color='red', label='Đường hồi quy')
plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.title("Hours vs Score")
plt.legend()
plt.show()

# PHẦN 9 – Đánh giá Model
from sklearn.metrics import r2_score
y_pred = model.predict(X)
score = r2_score(y, y_pred)
print(f"R2 Score: {score:.4f}")