import streamlit as st
import joblib
import pandas as pd
import os

# --- 1. การตั้งค่าหน้าเว็บ (UI/UX) ---
st.set_page_config(page_title="Steam Success Predictor", layout="wide")

st.title("🎮 Steam Success Predictor")
st.markdown("""
เครื่องมือนี้ใช้ **Artificial Intelligence** ในการคาดการณ์จำนวนเจ้าของเกมบน Steam 
โดยวิเคราะห์จากปัจจัยสำคัญ เช่น ราคา, จำนวนผู้เล่นพร้อมกัน และกระแสตอบรับจากรีวิว
""")

# --- 2. ฟังก์ชันโหลดโมเดลพร้อมระบบตรวจสอบไฟล์ (Error Handling) ---
@st.cache_resource
def load_my_model():
    model_path = 'steam_success_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

# พยายามโหลดโมเดล
model = load_my_model()

# --- 3. ส่วนรับข้อมูลด้านข้าง (Sidebar / Input Validation) ---
st.sidebar.header("📥 ข้อมูลปัจจัยของเกม")

with st.sidebar:
    price = st.number_input("ราคาเกม (USD)", min_value=0.0, value=9.99, 
                            help="ตั้งราคาขายของเกมในสกุลเงินดอลลาร์")
    ccu = st.number_input("จำนวนผู้เล่นพร้อมกัน (CCU)", min_value=0, value=100, 
                          help="Peak Concurrent Users")
    positive = st.number_input("จำนวนรีวิวบวก (Positive)", min_value=0, value=50)
    negative = st.number_input("จำนวนรีวิวลบ (Negative)", min_value=0, value=5)
    developer = st.text_input("ชื่อผู้พัฒนา (Developer)", value="Unknown")

# --- 4. ส่วนการแสดงผลการทำนาย ---
if st.button("🚀 วิเคราะห์และทำนายผล"):
    # ตรวจสอบก่อนว่าโหลดโมเดลสำเร็จหรือไม่ เพื่อป้องกัน NameError
    if model is not None:
        try:
            # เตรียมข้อมูลให้ตรงกับ format ของ Pipeline
            input_df = pd.DataFrame([{
                'price': price,
                'ccu': ccu,
                'positive': positive,
                'negative': negative,
                'developer': developer
            }])

            # ทำนายผล
            prediction = model.predict(input_df)[0]
            
            # ป้องกันค่าติดลบที่อาจเกิดจาก Linear Trend ในโมเดลบางตัว
            final_result = max(0, int(prediction))
            
            st.markdown("---")
            st.balloons() # เพิ่มลูกเล่นเมื่อทำนายสำเร็จ
            st.success(f"### คาดการณ์จำนวนเจ้าของเกม: {final_result:,} คน")
            
            # การแปลผลเชิง Business
            if final_result > 100000:
                st.info("💡 **วิเคราะห์:** เกมนี้มีศักยภาพสูงในการเป็นเกมยอดนิยม (Top Tier)")
            else:
                st.info("💡 **วิเคราะห์:** เกมนี้เหมาะกับกลุ่มเป้าหมายเฉพาะ (Niche Market)")
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
    else:
        # แสดงข้อความนี้แทน Error แดงๆ ถ้าหาไฟล์ .pkl ไม่เจอ
        st.error("❌ ไม่พบไฟล์โมเดล 'steam_success_model.pkl' ในระบบ กรุณาตรวจสอบตำแหน่งไฟล์บน GitHub")

# --- 5. ข้อมูลอธิบายตัวแปรและ Disclaimer ---
st.markdown("---")
with st.expander("ℹ️ ข้อมูลเพิ่มเติมเกี่ยวกับตัวแปร (Feature Description)"):
    st.write("""
    - **CCU (Concurrent Users):** จำนวนผู้เล่นที่ออนไลน์พร้อมกัน เป็นตัวชี้วัดความนิยมที่สำคัญที่สุด
    - **Positive/Negative Reviews:** พฤติกรรมการรีวิวสะท้อนถึงความพึงพอใจและคุณภาพของเกม
    - **Price:** ราคามีผลต่อการตัดสินใจซื้อในระดับที่แตกต่างกันตามประเภทเกม
    """)

st.warning("⚠️ **Disclaimer:** ผลการทำนายเป็นเพียงการประมาณการทางสถิติเพื่อการศึกษาเท่านั้น ไม่สามารถรับประกันยอดขายจริงได้")
