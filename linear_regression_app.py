import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Streamlit page setup
st.set_page_config(page_title="📈 Finance Linear Regression Dashboard", layout="wide")

# App Title
st.title("📊 Finance-Based Linear Regression App")

# Welcome GIF
st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExY3p1eXFianNoZ2p0NjF4NzUwZXkyZjZtanljaDE0cDJjaWZmM2o1dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/SEWEmCymjv8XDbsb8I/giphy.gif", width=1000)
st.markdown("""
Welcome to the **Finance Linear Regression Tool**!  
Upload a numeric dataset, explore it, and run a regression model to predict target variables such as stock prices, financial metrics, or other numeric values.

> 💡 Make sure your columns are numeric for regression to work correctly.
""")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your financial dataset (CSV only):", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.header("🔍 Dataset Overview")

    # Data preview
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Data description
    st.subheader("📊 Statistical Summary")
    st.write(df.describe())

    # Handle missing values
    st.subheader("🔧 Missing Values Check")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning("Missing values detected — automatically filling using forward fill and zeros.")
        df = df.fillna(method='ffill').fillna(0)
    else:
        st.success("No missing values detected!")

    # Filter numeric columns
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        st.error("Not enough numeric columns for regression.")
        st.stop()

    # Correlation heatmap
    st.subheader("📌 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Pairplot if dataset is small
    if numeric_df.shape[1] <= 5:
        st.subheader("🔍 Feature Relationships (Pairplot)")
        sns.pairplot(numeric_df)
        st.pyplot(plt.gcf())

    # Linear regression setup
    st.header("📈 Linear Regression Setup")

    target = st.selectbox("🎯 Select Target Variable:", numeric_df.columns)
    features = st.multiselect("📌 Select Independent Features (Predictors):", [col for col in numeric_df.columns if col != target])

    if len(features) > 0:
        X = numeric_df[features]
        y = numeric_df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.success("✅ Model trained successfully!")

        # Metrics
        st.subheader("📉 Model Evaluation Metrics")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")

        # Coefficients
        st.subheader("📊 Model Coefficients")
        coef_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        })
        st.dataframe(coef_df)

        # Interpretation
        st.subheader("🧠 Interpretation of Results")

        # R² interpretation
        if r2 >= 0.9:
            r2_msg = "Excellent fit — explains most of the variation in the target."
        elif r2 >= 0.7:
            r2_msg = "Good fit — captures a strong relationship."
        elif r2 >= 0.5:
            r2_msg = "Moderate fit — may be useful with caution."
        else:
            r2_msg = "Weak fit — predictions may not be reliable."

        st.markdown(f"**🔍 R² Interpretation:** {r2_msg}")
        st.markdown(f"**📉 MSE Interpretation:** The model's average squared prediction error is **{mse:.2f}**.")

        # Coefficient interpretations
        st.markdown("**📌 Feature Impact Analysis:**")
        for feature, coef in zip(features, model.coef_):
            trend = "increases" if coef > 0 else "decreases"
            st.write(f"- As **{feature}** increases, **{target}** tends to **{trend}** by **{abs(coef):.2f} units**.")

        # Interactive Plotly Visualization
        if len(features) == 1:
            st.subheader("📈 Interactive Plot: Actual vs Predicted with Trend Line")

            plot_df = pd.DataFrame({
                features[0]: X_test[features[0]],
                "Actual": y_test,
                "Predicted": y_pred
            }).sort_values(by=features[0])

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=plot_df[features[0]],
                y=plot_df["Actual"],
                mode='markers',
                name='Actual',
                marker=dict(color='blue'),
                hovertemplate=f"{features[0]}: %{{x}}<br>Actual: %{{y}}"
            ))

            fig.add_trace(go.Scatter(
                x=plot_df[features[0]],
                y=plot_df["Predicted"],
                mode='lines',
                name='Predicted (Trend Line)',
                line=dict(color='red'),
                hovertemplate=f"{features[0]}: %{{x}}<br>Predicted: %{{y}}"
            ))

            fig.update_layout(
                title="Actual vs Predicted",
                xaxis_title=features[0],
                yaxis_title=target,
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Trend line plot is only available when one feature is selected.")

    else:
        st.info("📌 Please select at least one feature to continue.")
else:
    st.info("📤 Upload your CSV file above to begin.")
