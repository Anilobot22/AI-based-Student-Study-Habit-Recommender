# -----------------------------------------------------------
# 🚀 StudyTrack AI - Batch 6 | Student Study Habit Recommender
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# --- PAGE CONFIG ---
st.set_page_config(page_title="Study Track AI Based Student Study Habit Recommender", layout="wide")

# --- SIDEBAR NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Home", "Model Training", "Data Insights", "Student Recommendation"])
st.sidebar.markdown("---")
st.sidebar.write("**Performance Dashboard**")

# ===========================================================
# 🌈 HOME PAGE
# ===========================================================
if page == "Home":
    page_bg = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #E3F2FD 10%, #FFF8E1 90%);
        color: #0d47a1;
    }
    [data-testid="stSidebar"] {
        background: #f5f5f5;
    }
    h1, h2, h3 {
        color: #0d47a1 !important;
    }
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

    st.title("🚀 AI Based Student Study Habit Recommender")
    st.markdown("### *Empowering Students through Predictive Analytics*")

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
        st.subheader("🎯 Precision")
        st.write("Predict marks using Random Forest Regression for accuracy and reliability.")

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/4140/4140037.png", width=120)
        st.subheader("🧠 Intelligence")
        st.write("AI learns from study patterns — StudyHours, SleepHours, AttentionLevel.")

    with col3:
        st.image("https://cdn-icons-png.flaticon.com/512/4140/4140048.png", width=120)
        st.subheader("📈 Growth")
        st.write("Customized recommendations to improve student performance.")

    st.divider()
    st.info("**Tip:** Go to 'Model Training' to upload your batch data and train the model.")

# ===========================================================
# 📂 MODEL TRAINING PAGE (Enhanced with Visualization)
# ===========================================================
elif page == "Model Training":
    st.header("📂 Train the AI Model")

    file_type = st.radio("Select Data Source:", ["CSV", "Excel"])
    uploaded_file = None

    if file_type == "CSV":
        uploaded_file = st.file_uploader("Upload Student Performance CSV", type=["csv"])
    elif file_type == "Excel":
        uploaded_file = st.file_uploader("Upload Student Performance Excel", type=["xlsx"])

    if uploaded_file is not None:
        # Load the uploaded file
        if file_type == "CSV":
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.success("✅ Data Loaded Successfully!")
        st.dataframe(data.head(10))

        # Split dataset into features (X) and target (y)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        if "Student" in X.columns:
            X = X.drop(columns=["Student"])

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Accuracy score
        score = model.score(X_test, y_test)

        st.success(f"🎯 Model Trained Successfully! Accuracy: {score * 100:.2f}%")

        # Combine actual vs predicted results into a DataFrame
        result_df = pd.DataFrame({
            "Actual Marks": y_test.values,
            "Predicted Marks": y_pred
        }).reset_index(drop=True)

        st.write("### 📊 Actual vs Predicted Marks (Test Data)")
        st.dataframe(result_df.head())

        # Plot line chart for comparison
        st.line_chart(result_df, use_container_width=True)

        # Save model and data to session state
        st.session_state["model"] = model
        st.session_state["data"] = data

# ===========================================================
# 📊 DATA INSIGHTS PAGE (Enhanced)
# ===========================================================
elif page == "Data Insights":
    st.header("📊 Data Insights Dashboard")

    if 'data' in st.session_state:
        df = st.session_state['data']
        st.success("✅ Data Loaded Successfully for Insights!")
        st.write("### 👀 Data Preview")
        st.dataframe(df.head())

        # -------------------------------------------------------
        # 🔹 1. Scatter Plot — StudyHours vs Marks
        # -------------------------------------------------------
        st.subheader("🎯 Study Hours vs Marks (Colored by Attention Level)")
        if "StudyHours" in df.columns and "Marks" in df.columns:
            fig = px.scatter(
                df,
                x="StudyHours",
                y="Marks",
                color="AttentionLevel" if "AttentionLevel" in df.columns else None,
                size="Exercise" if "Exercise" in df.columns else None,
                title="Relationship between Study Hours and Marks",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Columns 'StudyHours' and 'Marks' are required for this chart.")

        st.divider()

        # -------------------------------------------------------
        # 🔹 2. Bar Chart — Average Marks by StudyHours
        # -------------------------------------------------------
        st.subheader("📊 Average Marks by Study Hours")
        bar_df = df.groupby("StudyHours")["Marks"].mean().reset_index()
        fig_bar = px.bar(
            bar_df,
            x="StudyHours",
            y="Marks",
            color="Marks",
            title="Average Marks Achieved at Different Study Hours",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # -------------------------------------------------------
        # 🔹 3. Pie Chart — Attention Level Distribution
        # -------------------------------------------------------
        if "AttentionLevel" in df.columns:
            st.subheader("🥧 Attention Level Distribution")
            attention_counts = df["AttentionLevel"].value_counts().reset_index()
            attention_counts.columns = ["AttentionLevel", "Count"]
            fig_pie = px.pie(
                attention_counts,
                names="AttentionLevel",
                values="Count",
                color_discrete_sequence=px.colors.sequential.RdBu,
                title="Distribution of Attention Levels among Students"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Column 'AttentionLevel' not found for pie chart.")

        st.divider()

        # -------------------------------------------------------
        # 🔹 4. Heatmap — Correlation between Study Habits & Marks
        # -------------------------------------------------------
        st.subheader("🔥 Correlation Heatmap between Variables")
        num_df = df.select_dtypes(include=['int64', 'float64'])
        if not num_df.empty:
            corr = num_df.corr()
            fig_heatmap = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap (Closer to 1 or -1 means strong relation)"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("No numeric columns found for correlation analysis.")

        st.divider()

        # -------------------------------------------------------
        # 🔹 5. Top Insights Summary (AI-style Text Insights)
        # -------------------------------------------------------
        st.subheader("💡 Key Insights Summary")

        avg_study = df["StudyHours"].mean() if "StudyHours" in df.columns else 0
        avg_marks = df["Marks"].mean() if "Marks" in df.columns else 0
        top_student = df.loc[df["Marks"].idxmax(), "Student"] if "Student" in df.columns else "N/A"
        high_attention = df["AttentionLevel"].mean() if "AttentionLevel" in df.columns else 0

        st.markdown(f"""
        - 🧠 **Average Study Hours:** {avg_study:.2f} hrs/day  
        - 🎯 **Average Marks:** {avg_marks:.2f}%  
        - 👑 **Top Performing Student:** {top_student}  
        - 💹 **Average Attention Level:** {high_attention:.2f}  
        """)

        # Correlation between StudyHours & Marks
        if "StudyHours" in df.columns and "Marks" in df.columns:
            corr_val = df["StudyHours"].corr(df["Marks"])
            st.info(f"📈 **Correlation between Study Hours and Marks:** {corr_val:.2f}")
    else:
        st.warning("⚠️ Please train the model first in the 'Model Training' section.")

# ===========================================================
# 🎯 STUDENT RECOMMENDATION PAGE
# ===========================================================
elif page == "Student Recommendation":
    st.header("🎯 Student Recommendation System")

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first in 'Model Training' section.")
    else:
        model = st.session_state["model"]

        # =================== PART 1 : INDIVIDUAL PREDICTION ===================
        st.subheader("🧍 Individual Student Prediction")

        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Student Name")
            study = st.number_input("Study Hours", 0, 12)
            play = st.number_input("Play Hours", 0, 10)
        with col2:
            sleep = st.number_input("Sleep Hours", 0, 12)
            social = st.number_input("Social Media Hours", 0, 6)
        with col3:
            exercise = st.number_input("Exercise Hours", 0, 5)
            attention = st.slider("Attention Level (1–10)", 1, 10, 7)

        if st.button("Predict Individual Marks"):
            new_data = pd.DataFrame({
                "StudyHours": [study],
                "PlayHours": [play],
                "SleepHours": [sleep],
                "SocialMedia": [social],
                "Exercise": [exercise],
                "AttentionLevel": [attention]
            })

            predicted_mark = model.predict(new_data)[0]
            if predicted_mark > 100:
                predicted_mark = 100

            st.metric(label="Predicted Final Score", value=f"{predicted_mark:.1f}/100")

            # Recommendation logic
            if predicted_mark >= 85:
                rec = "🏆 Excellent! Keep up your great habits."
            elif predicted_mark >= 70:
                rec = "🧠 Good! Focus slightly more on weak areas."
            else:
                rec = "📘 Needs improvement. Increase study & reduce distractions."
            st.info(f"**Recommendation for {name if name else 'Student'}:** {rec}")

        st.divider()

        # =================== PART 2 : BULK PREDICTION ===================
        st.subheader("📊 Bulk Upload – Predict & Cluster Multiple Students")

        bulk_file = st.file_uploader("Upload Excel File for Bulk Prediction", type=["xlsx"])
        if bulk_file is not None:
            bulk_df = pd.read_excel(bulk_file)
            st.write("### Uploaded Student Data:")
            st.dataframe(bulk_df.head())

            # Ensure columns exist
            required_cols = ["StudyHours", "PlayHours", "SleepHours", "SocialMedia", "Exercise", "AttentionLevel"]
            if all(col in bulk_df.columns for col in required_cols):
                # Predict marks
                predicted_marks = model.predict(bulk_df[required_cols])
                bulk_df["Predicted_Marks"] = predicted_marks

                # KMeans clustering based on predicted marks
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                bulk_df["PerformanceCluster"] = kmeans.fit_predict(bulk_df[["Predicted_Marks"]])

                # Assign readable cluster names
                cluster_map = {0: "Low Performer", 1: "Average Performer", 2: "High Performer"}
                bulk_df["PerformanceLevel"] = bulk_df["PerformanceCluster"].map(cluster_map)


                # Generate recommendations
                def get_recommendation(mark):
                    if mark >= 85:
                        return "🏆 Excellent! Keep performing consistently."
                    elif mark >= 70:
                        return "🧠 Good! Slight improvement needed."
                    else:
                        return "📘 Needs improvement. Focus more on studies."


                bulk_df["Recommendation"] = bulk_df["Predicted_Marks"].apply(get_recommendation)

                st.success("✅ Bulk Prediction and Clustering Completed!")
                st.dataframe(bulk_df)

                # Plot cluster visualization
                fig = px.scatter(bulk_df, x="StudyHours", y="Predicted_Marks",
                                 color="PerformanceLevel", size="AttentionLevel",
                                 title="Student Clusters based on Predicted Marks")
                st.plotly_chart(fig, use_container_width=True)

                # Download result
                st.download_button(
                    label="📥 Download Predictions as Excel",
                    data=bulk_df.to_csv(index=False).encode('utf-8'),
                    file_name="Bulk_Predicted_Results.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ Uploaded file missing required columns.")

# ===========================================================
# 📜 FOOTER
# ===========================================================
st.markdown("---")
st.markdown(
    "<center>© 2025 StudyTrack AI | Designed for Batch-6 Students with ❤️ by Anil Kumar</center>",
    unsafe_allow_html=True
)
