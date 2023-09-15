
import librosa
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)



def train_emotion_model(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model



def predict_emotion(audio_file, model):
    features = extract_features(audio_file)
    emotion = model.predict([features])[0]
    return "happy" if emotion == 1 else "sad"



def calculate_talk_ratio(audio_file):

    return 0.7



def main():

    audio_files = ["happy_audio1.wav", "happy_audio2.wav", "sad_audio1.wav", "sad_audio2.wav"]
    emotions = [1, 1, 0, 0]  # 1 for happy, 0 for sad

    emotion_model = train_emotion_model([extract_features(file) for file in audio_files], emotions)


    results_emotion = []
    results_talk_ratio = []

    for audio_file in audio_files:
        predicted_emotion = predict_emotion(audio_file, emotion_model)
        predicted_talk_ratio = calculate_talk_ratio(audio_file)

        results_emotion.append((audio_file, predicted_emotion))
        results_talk_ratio.append((audio_file, predicted_talk_ratio))


    print("Predicted Emotions:")
    for audio_file, emotion in results_emotion:
        print(f"Audio File: {audio_file}, Emotion: {emotion}")


    plt.figure()
    for audio_file, emotion in results_emotion:
        plt.plot(audio_file, emotion)
    plt.xlabel("Audio File")
    plt.ylabel("Emotion (happy/sad)")
    plt.show()


    print("\nPredicted Talk Ratios:")
    for audio_file, talk_ratio in results_talk_ratio:
        print(f"Audio File: {audio_file}, Talk Ratio: {talk_ratio}")


    plt.figure()
    for audio_file, talk_ratio in results_talk_ratio:
        plt.plot(audio_file, talk_ratio)
    plt.xlabel("Audio File")
    plt.ylabel("Talk Ratio")
    plt.show()


if __name__ == "__main__":
    main()