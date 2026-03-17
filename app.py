import streamlit as st
import joblib
import pandas as pd
import os

# --- 1. การตั้งค่าหน้าเว็บ (UI/UX) ---
st.set_page_config(page_title="Steam Success Predictor", layout="wide")

st.title("🎮 Steam Success Predictor")
st.markdown("""
เครื่องมือนี้ใช้ **Artificial Intelligence (Machine Learning)** ในการคาดการณ์จำนวนเจ้าของเกมบน Steam 
โดยวิเคราะห์จากปัจจัยสำคัญ เช่น ราคา, จำนวนผู้เล่นพร้อมกัน และกระแสตอบรับจากรีวิว
""")

# --- 2. ฟังก์ชันโหลดโมเดลพร้อมระบบตรวจสอบไฟล์ (Error Handling) ---
@st.cache_resource
def load_my_model():
    # ตรวจสอบว่าไฟล์โมเดลอยู่ในตำแหน่งที่ถูกต้องหรือไม่
    model_path = 'steam_success_model.pkl' 
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
            return None
    else:
        return None

# พยายามโหลดโมเดลเก็บไว้ในตัวแปร global
model = load_my_model()

# --- 3. ส่วนรับข้อมูลด้านข้าง (Sidebar / Input Validation) ---
st.sidebar.header("📥 ข้อมูลปัจจัยของเกม")

with st.sidebar:
    price = st.number_input("ราคาเกม (USD)", min_value=0.0, value=9.99, step=0.01,
                            help="ตั้งราคาขายของเกมในสกุลเงินดอลลาร์")
    ccu = st.number_input("จำนวนผู้เล่นพร้อมกัน (CCU)", min_value=0, value=100, 
                          help="Peak Concurrent Users")
    positive = st.number_input("จำนวนรีวิวบวก (Positive)", min_value=0, value=50)
    negative = st.number_input("จำนวนรีวิวลบ (Negative)", min_value=0, value=5)
    developer = st.text_input("ชื่อผู้พัฒนา (Developer)", value="Unknown")

# --- 4. ส่วนการแสดงผลการทำนาย ---
if st.button("🚀 วิเคราะห์และทำนายผล"):
    # ป้องกัน NameError โดยตรวจสอบว่าโหลดโมเดลสำเร็จหรือไม่
    if model is not None:
        try:
            # เตรียมข้อมูลให้ตรงกับ format ของ Preprocessing Pipeline ในโค้ดเทรน
            input_df = pd.DataFrame([{
                'price': price,
                'ccu': ccu,
                'positive': positive,
                'negative': negative,
                'developer': developer
            }])

            # ทำนายผล (เรียกใช้ Predict จาก Pipeline ที่เซฟไว้)
            prediction = model.predict(input_df)[0]
            
            # ป้องกันค่าติดลบและจัดรูปแบบตัวเลข
            final_result = max(0, int(prediction))
            
            st.markdown("---")
            st.balloons() # เพิ่มเอฟเฟกต์เมื่อทำนายสำเร็จ
            st.success(f"### คาดการณ์จำนวนเจ้าของเกม: {final_result:,} คน")
            
            # การแปลผลเชิงธุรกิจเบื้องต้น
            if final_result > 100000:
                st.info("💡 **วิเคราะห์:** เกมนี้มีศักยภาพสูงในการเข้าถึงกลุ่มผู้เล่นวงกว้าง (Potential Hit)")
            else:
                st.info("💡 **วิเคราะห์:** เกมนี้มีแนวโน้มเข้าถึงกลุ่มผู้เล่นเฉพาะทาง (Niche Market)")
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")
    else:
        # แสดงข้อความแจ้งเตือนที่อ่านง่ายแทน Error ของระบบ
        st.error("❌ ไม่พบไฟล์โมเดล 'steam_success_model.pkl' ใน Repository กรุณาย้ายไฟล์ออกมาไว้ที่หน้าแรกสุด (Root)")

# --- 5. ส่วนอธิบายเพิ่มเติมและ Disclaimer ---
st.markdown("---")
with st.expander("ℹ️ ข้อมูลเพิ่มเติมเกี่ยวกับตัวแปร (Feature Meanings)"):
    st.write("""
    - **Price:** ราคามีผลต่อการตัดสินใจซื้อในระดับที่แตกต่างกันตามคุณภาพเกม
    - **CCU (Concurrent Users):** จำนวนผู้เล่นที่ออนไลน์พร้อมกัน สะท้อนความนิยมแบบ Real-time
    - **Reviews:** พฤติกรรมการรีวิวสะท้อนถึงความพึงพอใจและกระแสบอกต่อ (Word of Mouth)
    """)

st.warning("⚠️ **Disclaimer:** ผลการทำนายเป็นเพียงการประมาณการทางสถิติจากข้อมูลในอดีตเท่านั้น ไม่สามารถการันตียอดขายจริงได้ 100%")
