import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import matplotlib.pyplot as plt


st.set_page_config(page_title="Web Application for Malignant Risk Prediction in Parotid Tumors (PTs)", layout="wide")


@st.cache_resource
def load_model():
    model = load('LightGBM.pkl')
    return model

model = load_model()


st.title("Web Application for Malignant Risk Prediction in Parotid Tumors (PTs)")
st.write("Please input patient's clinical indicators:")


col1 = st.columns(1)[0]

with col1:
    # 修改输入控件，添加标签
    with col1:
        st.markdown("**Age (years)**")
        age = st.number_input('Age input', value=50, format="%d", label_visibility="collapsed")
        st.write("")  
        
        st.markdown("**Shape**")
        shape = st.selectbox('Shape input', [0, 1], format_func=lambda x: "Regular" if x == 0 else "Irregular", label_visibility="collapsed")
        st.write("")  
    
    with col1:
        st.markdown("**Boundary**")
        boundary = st.selectbox('Boundary input', [0, 1], format_func=lambda x: "Clear" if x == 0 else "Unclear", label_visibility="collapsed")
        st.write("")  
        
        st.markdown("**Number**")
        number = st.selectbox('Number input', [0, 1], format_func=lambda x: "Single" if x == 0 else "Multiple", label_visibility="collapsed")
        st.write("")  
        
        st.markdown("**Location**")
        position = st.selectbox('Location input', [0, 1], format_func=lambda x: "Superficial" if x == 0 else "Deep or Both", label_visibility="collapsed")
        st.write("")  
        
        st.markdown("**Lymph Metastasis (LM)**")
        lm = st.selectbox('LM input', [0, 1], format_func=lambda x: "Absent" if x == 0 else "Present", label_visibility="collapsed")
        st.write("")  
        
        st.markdown("**DL_sign**")
        dl_sign = st.number_input('DL_sign input', min_value=0.0, max_value=1.0, value=0.5, format="%.3f", help="Probability of the DL model", label_visibility="collapsed")
        st.write("")  


if st.button('Predict'):

    input_data = pd.DataFrame({
        'Age': [age],
        'Shape': [shape],
        'Number': [number],
        'Boundary': [boundary],
        'Position': [position],
        'LM': [lm],
        'DL_sign': [dl_sign]
    })


    prediction = model.predict_proba(input_data)[0]
    

    st.write("---")
    st.subheader("Prediction Results")
    

    st.metric(
        label="Probability of Malignancy",
        value=f"{prediction[1]:.1%}",
        delta=None
    )
    
    # 修改风险等级显示
    risk_level = "High Risk" if prediction[1] > 0.5 else "Low Risk"
    risk_message = "High Risk: Please consult a specialist promptly." if prediction[1] > 0.5 else "Low Risk: Please schedule regular reviews or consider other clinically relevant examinations."
    st.info(f"Risk Level: {risk_level}\n\n{risk_message}")

    st.write("---")
    st.subheader("Model Interpretation")
    
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # 绘制force plot
    plt.figure(figsize=(15, 6))  # 增加高度
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_data.iloc[0],
        matplotlib=True,
        show=False,
        text_rotation=0,
        contribution_threshold=0.05,  # 调整贡献阈值
        plot_cmap=['#FF4B4B', '#4B4BFF']  # 使用更鲜艳的颜色
    )
    plt.gcf().set_facecolor('white')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  # 调整四周边距
    st.pyplot(plt)
    plt.close()

    st.write("---")
    st.subheader("Feature Contribution Analysis")
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'Feature': input_data.columns,
        'SHAP Value': np.abs(shap_values[0])
    }).sort_values('SHAP Value', ascending=False)
    
    # 显示特征重要性表格
    st.table(feature_importance)


st.write("---")
st.markdown("""
### Instructions:
1. Enter the patient's clinical indicators in the input fields above
2. Click the "Predict" button to get results
3. The system will display the probability of malignancy and risk level
4. SHAP values show how each feature contributes to the prediction
""")


st.sidebar.title("Model Information")
st.sidebar.info("""
- Model Type: LightGBM Classifier
- Training Data: Parotid Tumors Clinical Data
- Target Variable: Malignancy Classification
- Number of Features: 6 Clinical Indicators
""")


st.sidebar.title("Feature Description")
st.sidebar.markdown("""
- Age: Patient's age (years)
- Shape: Lesion shape (0: Regular / 1: Irregular)
- Boundary: Lesion boundary (0: Clear / 1: Unclear)
- Number: Lesion count (0: Single / 1: Multiple)
- Location: Lesion position (0: Superficial / 1: Deep or Both)
- LM: Lymph Metastasis (0: Absent / 1: Present)
- DL_sign: Probability of the DL model (0-1)
""")