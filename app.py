import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. การตั้งค่าหน้าเว็บ (UI/UX) ---
st.set_page_config(page_title="Steam Success Predictor", layout="wide")

st.title("🎮 Steam Success Predictor")
st.markdown("""
เครื่องมือนี้ใช้ **Artificial Intelligence (Machine Learning)** ในการคาดการณ์จำนวนเจ้าของเกมบน Steam 
โดยวิเคราะห์จากปัจจัยสำคัญ เช่น ราคา, จำนวนผู้เล่นพร้อมกัน และกระแสตอบรับจากรีวิว
""")

# --- 2. โหลดโมเดลที่ Save ไว้ ---
@st.cache_resource # ใช้แคชเพื่อความรวดเร็วในการโหลดเว็บ
def load_my_model():
    return joblib.load('steam_success_model.pkl')

try:
    model = load_my_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'steam_success_model.pkl' กรุณาตรวจสอบใน Repository")

# --- 3. ส่วนรับข้อมูล (Input Validation) ---
st.sidebar.header("📥 ข้อมูลปัจจัยของเกม")

with st.sidebar:
    # กำหนด min_value เพื่อป้องกันค่าที่ไม่สมเหตุสมผล (Input Validation)
    price = st.number_input("ราคาเกม (USD)", min_value=0.0, max_value=1000.0, value=9.99, 
                            help="ตั้งราคาขายของเกมในสกุลเงินดอลลาร์")
    
    ccu = st.number_input("จำนวนผู้เล่นพร้อมกัน (CCU)", min_value=0, value=100, 
                          help="Peak Concurrent Users ที่คาดหวังหรือเป็นอยู่ในปัจจุบัน")
    
    positive = st.number_input("จำนวนรีวิวบวก (Positive)", min_value=0, value=50)
    
    negative = st.number_input("จำนวนรีวิวลบ (Negative)", min_value=0, value=5)
    
    developer = st.text_input("ชื่อผู้พัฒนา (Developer)", value="Unknown", 
                              help="ชื่อบริษัทผู้พัฒนาเกม")

# --- 4. การแสดงผลการทำนาย (Prediction) ---
if st.button("🚀 วิเคราะห์และทำนายผล"):
    # เตรียมข้อมูลให้ตรงกับ Format ที่ Pipeline ต้องการ
    input_df = pd.DataFrame([{
        'price': price,
        'ccu': ccu,
        'positive': positive,
        'negative': negative,
        'developer': developer
    }])

    # ทำนายผล
    prediction = model.predict(input_df)[0]
    
    # แสดงผลลัพธ์
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 ผลการคาดการณ์")
        # แสดงผลลัพธ์ให้ชัดเจนพร้อมตัวเลขขั้นหลักพัน
        st.metric(label="จำนวนเจ้าของเกมที่คาดหวัง (Estimated Owners)", value=f"{int(prediction):,} คน")
        
    with col2:
        st.subheader("💡 คำแนะนำเบื้องต้น")
        if prediction > 100000:
            st.success("เกมของคุณมีแนวโน้มที่จะเป็นเกมกระแสหลัก (Mainstream Success)")
        else:
            st.info("เกมของคุณมีขนาดกลุ่มเป้าหมายเฉพาะเจาะจง (Niche Market)")

# --- 5. คำอธิบาย Features (เพื่อให้ผู้ใช้ที่ไม่รู้จัก ML เข้าใจ) ---
with st.expander("ℹ️ ข้อมูลเพิ่มเติมเกี่ยวกับตัวแปร (Feature Meanings)"):
    st.write("""
    - **CCU:** ย่อมาจาก Concurrent Users คือจำนวนคนที่ออนไลน์เล่นเกมพร้อมกัน เป็นตัวบ่งชี้ความนิยมสูงสุด
    - **Positive/Negative Reviews:** กระแสตอบรับจากผู้เล่นจริงที่ส่งผลต่ออัลกอริทึมของ Steam
    - **Price:** ราคาที่มีผลต่อการตัดสินใจซื้อและการเข้าถึงกลุ่มเป้าหมาย
    """)

# --- 6. Disclaimer (ตามเกณฑ์คะแนน) ---
st.markdown("---")
st.warning("⚠️ **Disclaimer:** ผลการทำนายนี้เป็นเพียงการประมาณการทางสถิติจากข้อมูลในอดีตเท่านั้น ไม่สามารถการันตียอดขายจริงได้ 100% โปรดใช้ประกอบการตัดสินใจทางธุรกิจร่วมกับปัจจัยอื่นๆ")
