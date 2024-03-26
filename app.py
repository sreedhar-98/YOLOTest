import streamlit as st
from streamlit_webrtc import webrtc_streamer
from callback import callback
from twilio.rest import Client
from ultralytics import YOLO



@st.cache_resource()
def get_detector():
    return YOLO("yolov8n_openvino_model/",task="detect",verbose=False)



@st.cache_data(ttl=86400)
def get_twilio_token():
    account_sid=st.secrets["sid"]
    auth_token=st.secrets["token"]
    client=Client(account_sid,auth_token)
    token=client.tokens.create()
    return token


token=get_twilio_token()
model=get_detector()

st.title("Person Counter App")

webrtc_streamer(key="sample", video_frame_callback=lambda frame:callback(frame,model),media_stream_constraints={"video":True,"audio":False},
                 rtc_configuration={"iceServers":token.ice_servers})