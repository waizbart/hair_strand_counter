import cv2

# Carregar a imagem
imagem = cv2.imread('hair2.png')

# Converter a imagem para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicar um detector de bordas (por exemplo, Canny)
bordas = cv2.Canny(imagem_cinza, threshold1=100, threshold2=200)

cv2.imshow('Bordas', bordas)
cv2.waitKey(0)

# Encontrar contornos na imagem
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Contar o número de contornos (fios de cabelo)
quantidade_fios = len(contornos)

print(f"Quantidade de fios de cabelo: {quantidade_fios}")

# Desenhar os contornos na imagem original
cv2.drawContours(imagem, contornos, -1, (0, 255, 0), 2)

# Exibir a imagem com os contornos destacados
cv2.imshow('Imagem com Contornos', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
