import cv2
import mediapipe as mp

mp_maos = mp.solutions.hands
maos = mp_maos.Hands()
mp_desenho = mp.solutions.drawing_utils

LETRAS = {
    "A": {
        "polegar": False,
        "indicador": False,
        "medio": False,
        "anelar": False,
        "mindinho": False,
        "curvado": False,
    },
    "B": {
        "polegar": True,
        "indicador": True,
        "medio": True,
        "anelar": True,
        "mindinho": True,
        "curvado": False,
    },
    "C": {
        "polegar": True,
        "indicador": True,
        "medio": True,
        "anelar": True,
        "mindinho": True,
        "curvado": True,
    },
    "D": {
        "polegar": True,
        "indicador": False,
        "medio": False,
        "anelar": False,
        "mindinho": False,
        "curvado": False,
    },
    "L": {
        "polegar": True,
        "indicador": True,
        "medio": False,
        "anelar": False,
        "mindinho": False,
        "curvado": False,
    },
}


def verifica_dedo_levantado(landmarks, dedo):
    dedos = {
        "polegar": {"ponta": 4, "meio": 3, "base": 2},
        "indicador": {"ponta": 8, "meio": 7, "base": 6},
        "medio": {"ponta": 12, "meio": 11, "base": 10},
        "anelar": {"ponta": 16, "meio": 15, "base": 14},
        "mindinho": {"ponta": 20, "meio": 19, "base": 18},
    }
    ponta = landmarks.landmark[dedos[dedo]["ponta"]]
    base = landmarks.landmark[dedos[dedo]["base"]]

    # Verifica se a ponta está acima da base com uma margem de tolerância
    return ponta.y < base.y - 0.02


def verifica_dedo_curvado(landmarks, dedo):
    dedos = {
        "polegar": {"ponta": 4, "meio": 3, "base": 2},
        "indicador": {"ponta": 8, "meio": 7, "base": 6},
        "medio": {"ponta": 12, "meio": 11, "base": 10},
        "anelar": {"ponta": 16, "meio": 15, "base": 14},
        "mindinho": {"ponta": 20, "meio": 19, "base": 18},
    }
    if dedo not in dedos:
        return False  # Retorna False se o dedo não for válido

    ponta = landmarks.landmark[dedos[dedo]["ponta"]]
    meio = landmarks.landmark[dedos[dedo]["meio"]]
    base = landmarks.landmark[dedos[dedo]["base"]]

    # Verifica se o dedo está curvado (a ponta está mais próxima da base do que do meio)
    return abs(ponta.y - base.y) < abs(ponta.y - meio.y)


for letara, requisitos in LETRAS.items():
    match = all(
        (estado_dedos[d] == requisitos[d] for d in requisitos if d != "curvado")
        for d in requisitos
    )
    if match:
        print(f"Letra: {letara}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        break

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

historico_estados = []


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
                "mindinho": verifica_dedo_levantado(landmarks, "mindinho"),
            }

            print("Estado dos dedos:", estado_dedos)

            historico_estados.append(estado_dedos)
            if len(historico_estados) > 10:  # Use os últimos 10 quadros
                historico_estados.pop(0)

            # Verifica a consistência dos estados
            estado_medio = {
                dedo: sum(h[dedo] for h in historico_estados)
                > len(historico_estados) // 2
                for dedo in estado_dedos
            }

            for letra, requisitos in LETRAS.items():
                match = all(
                    (
                        estado_medio[d] == requisitos[d]
                        if d != "curvado"
                        else requisitos[d] == verifica_dedo_curvado(landmarks, d)
                    )
                    for d in requisitos
                )
                if match:
                    cv2.putText(
                        frame,
                        f"Letra: {letra}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    break

            print(
                "Curvatura dos dedos:",
                {dedo: verifica_dedo_curvado(landmarks, dedo) for dedo in estado_dedos},
            )

    cv2.imshow("Reconhecimento de Letras", frame)

    key = cv2.waitKey(1)
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
