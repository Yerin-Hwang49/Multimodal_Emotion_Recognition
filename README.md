# Multimodal_Emotion_Recognition

본 연구에서는 빠른 반응과 정확한 이해를 조화시키는 이러한 사람의 감정 분석 과정을 모방하여, Fast-Thinking Classifier와 Slow-Thinking Classifier로 구성된 모델을 제안한다. 먼저, Fast-Thinking Classifier는 텍스트와 음성 데이터를 인풋으로 하여 발화가 중립인지 아닌지를 분류한다. 만약 중립이 아니라고 판단될 경우, Slow-Thinking classifier가 텍스트와 음성 데이터에 더해 피부 전도도(EDA), 체온(Temperature), 심장 박동 데이터(IBI)를 활용하여 발화에서 확인할 수 있는 감정을 분노, 혐오, 공포, 기쁨, 슬픔, 놀람의 감정으로 상세하고 정확하게 분류한다.
