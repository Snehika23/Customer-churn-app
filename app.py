import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Customer Churn Prediction")
st.markdown("Upload the **Telco Customer Churn CSV**, train the model, then predict churn for any customer.")

# ── Session state ─────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "encoders" not in st.session_state:
    st.session_state.encoders = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

# ── Helper: preprocess ────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

    object_columns = df.select_dtypes(include="object").columns
    encoders = {}
    for col in object_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📂 Data & EDA", "🤖 Train Model", "🔮 Predict"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Data Upload & EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload Telco-Customer-Churn.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
        st.session_state.df_raw = df
        st.success(f"✅ Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Info")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        with col2:
            st.subheader("Class Distribution")
            churn_counts = df["Churn"].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(["No Churn", "Churn"], churn_counts.values,
                   color=["#4CAF50", "#F44336"])
            ax.set_ylabel("Count")
            ax.set_title("Churn Distribution")
            for i, v in enumerate(churn_counts.values):
                ax.text(i, v + 30, str(v), ha="center", fontweight="bold")
            st.pyplot(fig)
            plt.close()

        st.subheader("Exploratory Data Analysis")
        eda_choice = st.selectbox("Choose a chart", [
            "Distribution — tenure",
            "Distribution — MonthlyCharges",
            "Distribution — TotalCharges",
            "Correlation Heatmap",
            "Countplot — Contract",
            "Countplot — InternetService",
            "Countplot — PaymentMethod",
        ])

        fig, ax = plt.subplots(figsize=(7, 3.5))

        if eda_choice.startswith("Distribution"):
            col_name = eda_choice.split("— ")[1]
            df_tmp = df.copy()
            df_tmp["TotalCharges"] = pd.to_numeric(
                df_tmp["TotalCharges"].replace({" ": "0.0"}), errors="coerce"
            ).fillna(0)
            sns.histplot(df_tmp[col_name], kde=True, ax=ax)
            ax.axvline(df_tmp[col_name].mean(), color="red",
                       linestyle="--", label=f"Mean: {df_tmp[col_name].mean():.1f}")
            ax.axvline(df_tmp[col_name].median(), color="green",
                       linestyle="-", label=f"Median: {df_tmp[col_name].median():.1f}")
            ax.set_title(f"Distribution of {col_name}")
            ax.legend()

        elif eda_choice == "Correlation Heatmap":
            df_tmp = df.copy()
            df_tmp["TotalCharges"] = pd.to_numeric(
                df_tmp["TotalCharges"].replace({" ": "0.0"}), errors="coerce"
            ).fillna(0)
            sns.heatmap(
                df_tmp[["tenure", "MonthlyCharges", "TotalCharges"]].corr(),
                annot=True, cmap="coolwarm", fmt=".2f", ax=ax
            )
            ax.set_title("Correlation Heatmap")

        else:
            col_name = eda_choice.split("— ")[1]
            order = df[col_name].value_counts().index
            sns.countplot(x=df[col_name], order=order, ax=ax)
            ax.set_title(f"Count Plot — {col_name}")
            plt.xticks(rotation=20, ha="right")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.info("👆 Upload the CSV file to get started.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Train Model
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Train Random Forest Model")

    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload the dataset in the **Data & EDA** tab first.")
    else:
        st.markdown("The model uses **Random Forest** with **SMOTE** to handle class imbalance, identical to your Colab notebook.")

        col_a, col_b = st.columns(2)
        with col_a:
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
            n_estimators = st.slider("Number of trees", 50, 300, 100, 50)
        with col_b:
            cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
            random_state = st.number_input("Random state", value=42)

        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Preprocessing data..."):
                df_proc, encoders = preprocess(st.session_state.df_raw)
                X = df_proc.drop(columns=["Churn"])
                y = df_proc["Churn"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state)
                )

            with st.spinner("Applying SMOTE..."):
                smote = SMOTE(random_state=int(random_state))
                X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

            with st.spinner(f"Training Random Forest ({n_estimators} trees)..."):
                rfc = RandomForestClassifier(
                    n_estimators=n_estimators, random_state=int(random_state)
                )
                scores = cross_val_score(rfc, X_train_sm, y_train_sm,
                                         cv=cv_folds, scoring="accuracy")
                rfc.fit(X_train_sm, y_train_sm)

            # Save to session state
            st.session_state.model = rfc
            st.session_state.encoders = encoders
            st.session_state.feature_names = X.columns.tolist()

            # Results
            y_pred = rfc.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred, output_dict=True)

            st.success("✅ Model trained successfully!")

            m1, m2, m3 = st.columns(3)
            m1.metric("Test Accuracy", f"{acc*100:.2f}%")
            m2.metric("CV Accuracy (mean)", f"{np.mean(scores)*100:.2f}%")
            m3.metric("CV Std Dev", f"±{np.std(scores)*100:.2f}%")

            col_r, col_c = st.columns(2)

            with col_r:
                st.subheader("Classification Report")
                report_df = pd.DataFrame(cr).transpose().round(2)
                st.dataframe(report_df, use_container_width=True)

            with col_c:
                st.subheader("Confusion Matrix")
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["No Churn", "Churn"],
                            yticklabels=["No Churn", "Churn"], ax=ax2)
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Actual")
                ax2.set_title("Confusion Matrix")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

            st.subheader("Feature Importances (Top 15)")
            feat_imp = pd.Series(rfc.feature_importances_,
                                 index=X.columns).sort_values(ascending=False).head(15)
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            feat_imp.plot(kind="barh", ax=ax3, color="#1976D2")
            ax3.invert_yaxis()
            ax3.set_xlabel("Importance")
            ax3.set_title("Top 15 Feature Importances")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Predict
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Predict Churn for a Customer")

    if st.session_state.model is None:
        st.warning("⚠️ Please train the model in the **Train Model** tab first.")
    else:
        st.markdown("Fill in the customer details below and click **Predict**.")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Personal Info")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1],
                                  format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines",
                                          ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service",
                                            ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security",
                                           ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup",
                                         ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection",
                                             ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support",
                                        ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV",
                                        ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies",
                                            ["Yes", "No", "No internet service"])

        with c3:
            st.subheader("Billing")
            contract = st.selectbox("Contract",
                                    ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 29.85, 0.5)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                            float(tenure * monthly_charges), 1.0)

        if st.button("🔮 Predict Churn", type="primary"):
            input_data = {
                "customerID": "new-customer",
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            input_df = pd.DataFrame([input_data])
            encoders = st.session_state.encoders

            for col, enc in encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = enc.transform(input_df[col].astype(str))
                    except ValueError:
                        # unseen label — use most frequent class
                        input_df[col] = enc.transform([enc.classes_[0]])[0]

            # Keep only training features in the right order
            input_df = input_df[st.session_state.feature_names]

            model = st.session_state.model
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]

            st.markdown("---")
            if prediction == 1:
                st.error(f"## ⚠️ This customer is likely to CHURN")
            else:
                st.success(f"## ✅ This customer is likely to STAY")

            r1, r2 = st.columns(2)
            r1.metric("Churn Probability", f"{prob[1]*100:.1f}%")
            r2.metric("Stay Probability", f"{prob[0]*100:.1f}%")

            # Probability bar
            fig4, ax4 = plt.subplots(figsize=(5, 1.2))
            ax4.barh(["Churn risk"], [prob[1]], color="#F44336" if prob[1] > 0.5 else "#FFC107")
            ax4.barh(["Churn risk"], [prob[0]], left=[prob[1]], color="#4CAF50")
            ax4.set_xlim(0, 1)
            ax4.set_xlabel("Probability")
            ax4.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
            ax4.set_title("Churn vs Stay probability")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        st.markdown("---")
        st.subheader("Batch Predict from CSV")
        batch_file = st.file_uploader("Upload a CSV of customers (same columns, no Churn column needed)",
                                      type=["csv"], key="batch")
        if batch_file:
            batch_df = pd.read_csv(batch_file)
            batch_input = batch_df.copy()
            for col, enc in st.session_state.encoders.items():
                if col in batch_input.columns:
                    batch_input[col] = batch_input[col].astype(str).apply(
                        lambda x: enc.transform([x])[0]
                        if x in enc.classes_ else enc.transform([enc.classes_[0]])[0]
                    )
            batch_input["TotalCharges"] = pd.to_numeric(
                batch_input["TotalCharges"].replace({" ": "0.0"}), errors="coerce"
            ).fillna(0)
            batch_input = batch_input[st.session_state.feature_names]
            preds = st.session_state.model.predict(batch_input)
            probs = st.session_state.model.predict_proba(batch_input)[:, 1]
            batch_df["Prediction"] = np.where(preds == 1, "Churn", "No Churn")
            batch_df["Churn Probability"] = (probs * 100).round(1)
            st.dataframe(batch_df[["customerID", "Prediction", "Churn Probability"]
                                   if "customerID" in batch_df.columns
                                   else ["Prediction", "Churn Probability"]],
                         use_container_width=True)
            csv_out = batch_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", csv_out,
                               "churn_predictions.csv", "text/csv")
