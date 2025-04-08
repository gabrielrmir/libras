import cv2
import mediapipe as mp

mp_maos = mp.solutions.hands
maos = mp_maos.Hands()
mp_desenho = mp.solutions.drawing_utils

LETRAS = {
    "A": {"polegar": False, "indicador": False, "medio": False, "anelar": False, "mindinho": False},
    "B": {"polegar": True, "indicador": True, "medio": True, "anelar": True, "mindinho": True},
    "C": {"polegar": True, "indicador": True, "medio": True, "anelar": True, "mindinho": True, "curvado": True},
    "D": {"polegar": True, "indicador": False, "medio": False, "anelar": False, "mindinho": False},
    "L": {"polegar": True, "indicador": True, "medio": False, "anelar": False, "mindinho": False}
}

def verifica_dedo_levantado(landmarks, dedo):

    dedos = {
        "polegar": {"ponta": 4, "base": 1},
        "indicador": {"ponta": 8, "base": 5},
        "medio": {"ponta": 12, "base": 9},
        "anelar": {"ponta": 16, "base": 13},
        "mindinho": {"ponta": 20, "base": 17}
    }
    return landmarks.landmark[dedos[dedo]["ponta"]].y < landmarks.landmark[dedos[dedo]["base"]].y


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = maos.process(frame_rgb)
    
    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)
            

            estado_dedos = {
                "polegar": verifica_dedo_levantado(landmarks, "polegar"),
                "indicador": verifica_dedo_levantado(landmarks, "indicador"),
                "medio": verifica_dedo_levantado(landmarks, "medio"),
                "anelar": verifica_dedo_levantado(landmarks, "anelar"),
                "mindinho": verifica_dedo_levantado(landmarks, "mindinho")
            }
            
          
            for letra, requisitos in LETRAS.items():
                match = all(estado_dedos[d] == requisitos[d] for d in requisitos if d != "curvado")
                if match:
                    cv2.putText(frame, f"Letra: {letra}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    break
    
    cv2.imshow('Reconhecimento de Letras', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()