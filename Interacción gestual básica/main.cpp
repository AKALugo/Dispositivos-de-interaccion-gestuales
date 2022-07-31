// Práctica realizada por Alejandro Lugo Fumero y Joseph Francisco Gabino Rodríguez.

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

double angle(Point s, Point e, Point f) {

	// Vectores entre el f y s, f y e.
    double v1[2],v2[2];
    v1[0] = s.x - f.x;
    v1[1] = s.y - f.y;
    v2[0] = e.x - f.x;
    v2[1] = e.y - f.y;
	// Calculamos los ángulos mediante la arcotangente (respecto a la horizontal).
    double ang1 = atan2(v1[1], v1[0]);
    double ang2 = atan2(v2[1], v2[0]);

	// Si se la de rango entre pi y -pi, la devolvemos al rango.
    double ang = ang1 - ang2;
    if (ang > CV_PI) ang -= 2*CV_PI;
    if (ang < -CV_PI) ang += 2*CV_PI;

	// Devolvemos el angulo en grados, arcotangente la devuelve en radianes.
    return ang*180/CV_PI;
}

int main(int argc, char* argv[])
{
	// Nuestras imagenes
	Mat frame, roi, fgMask;
	// Captura la imagen a traves de la webcam
	VideoCapture cap;
	// Abre la capturadora, se usa 0 para que pille la que viene por defecto
	cap.open(0);

	// Si la webcam no se ha abierto
	if (!cap.isOpened()) {
		printf("Error opening cam\n");
		return -1;
	}

	// Puntero al metodo que nos proporciona la máscara binaria (MOG2).
	Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();

	// Bool que sirve para cambiar el learning rate a 0 cuando el algoritmo haya aprendido bien
	// que parte es el fondo.
	bool learningRate = false;

	// Vector de vectores de puntos, un contorno no es más que un vector de puntos, pero findcontours
	// detecta todos los contornos por eso hay que pasarle todos los conotornos a la vez.
	vector<vector<Point> > contours;


	// Crea ventanas.
	namedWindow("Frame");
	namedWindow("ROI");
	namedWindow("Foreground Mask");
	
	// Rectangulo 400 pixeles a la derecha, 100 para abajo y un rectangulo de 200x200 pixeles
	// Se empieza a mirar desde la esquina superior izquierda
	Rect rect(400,100,200,200);
	// Rectangulo de area minima que engloba completamente nuestro contorno.
	Rect boundRect;

	// Guarda el indice en el vector de vectores de punto de la mano.
	int mano;
	// Mostrar gesto por la pantalla y la información.
	string texto = "", info = "";
	// Int para contar los defectos y poder saber que gesto estamos haciendo.
	int contador;
	// Ventor para guardar los puntos Start y End de los defectos de convexidad.
	vector<pair<Point, Point>> SEvector;

	// Bucle infinito
	while(true) {

        // Captura imagenes gracias a nuestra capturadora
		cap >> frame;
		contador = 0;

		// Hacemos un flip horizontal de la imagen, para que si movemos la mano a la derecha
		// Veremos nuestra mano moverse asia la derecha en la cámara (así no nos equivocamos)
		flip(frame,frame,1);

		// De frame seleccionamos el rectangulo definido en rect y sacamos esa región en roi
		// Hacemos una copia para que si modificamos algo en roi no afecte a frame.
		frame(rect).copyTo(roi);

		// Método que de nuestra imagen actual nos devuelve la máscara de lo que está delante del fondo
		// Esto lo hace mediante un 3 argumento que por defecto vale -1 y le indica como de rápido olvida
		// lo nuevo que aparece en la escena y lo integra como parte del fondo (este valor va entre 0 y 1)
		if (!learningRate) {
			pBackSub->apply(roi, fgMask);
			info = "Capturando fondo";
		} 
		else {
			pBackSub->apply(roi, fgMask, 0);
			info = "Fondo capturado";
		}			   


		// Detecta los contornos, contours para pasarle todos los contornos, RETR_EXTERNAL para solo los contornos externos
		// y CHAIN_APPROX_SIMPLE para aproximar los píxeles vecinos al contorno.
		findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		// Pintamos todos los contornos, -1 para que pinte todos los contornos y Scalar para el color del contorno
		// y el ancho, en este caso 3 píxeles.
		drawContours(roi, contours, -1, Scalar(0,255,0),3);
			
		mano = 0;
		// Bucle que busca el contorno de mayor tamaño, por defecto asumimos que el mayor es la posicion 0.
		for (int i = 1; i < contours.size(); i++)
			if (contours[mano].size() < contours[i].size())
				mano = i;

		// Vector para los índices del contorno.
		vector<int> hull;
		// Método que convierte un vector de puntos en los índices sobre el contorno.
		// Primer false si queremos sentido horario o no y el segundo false para si queremos
		// los puntos o los índices del contorno (en este caso queremos los índices).
		convexHull(contours[mano], hull,false,false);
		// Ordenamos los indices para que los contornos sean monotonamente creciente.
		sort(hull.begin(),hull.end(),greater <int>());

		// Vector de Vec4i que almacena 4 valores de enteros.
		vector<Vec4i> defects;
		// Sacamos los defectos de convexidad.
		convexityDefects(contours[mano], hull, defects);
		for (int i = 0; i < defects.size(); i++) {
			Point s = contours[mano][defects[i][0]];
			Point e = contours[mano][defects[i][1]];
			Point f = contours[mano][defects[i][2]];
			// Devuelve valor entero pero lo queremos en píxeles, por eso / 256.0
			float depth = (float)defects[i][3] / 256.0;
			// Ángulo que forman el punto Start y End.
			double ang = angle(s,e,f);
			
			// Si forma un angulo menor o igual de 90 y profundidad mayor a 20 píxeles.
			if (depth > 20.0 && ang <= 90) {
				contador ++;
				// Pintamos los defectos de convexidad
				circle(roi, f,5,Scalar(0,0,255),-1);
				// Pintamos la malla
				line(roi,s,e,Scalar(255,0,0),2);

				// Guardamos en nuestro vector los puntos Start y End.
				SEvector.push_back(make_pair(s,e));
			}
		}

		// Englobamos nuestro contorno.
		boundRect = boundingRect(contours[mano]);
		rectangle(roi,boundRect,Scalar(0,0,255),3);

		// Si el area que engloba nuestro contorno es mayor que 50x50 es que nuestra mano está en la zona y 
		// a partir de ahí mostramos el texto con la ayuda de un contador de defectos de convexidad.
		if (boundRect.height > 50 && boundRect.width > 50 && learningRate)
			switch (contador) {

			case 0:
				// Si es casi igual de ancho que de alto.
				if (abs(boundRect.height - boundRect.width) < 38) texto = "Numero 0";
				// Si es más ancho que alto.
				else if (boundRect.width > boundRect.height) texto = "Beber";
				// Si es más alto que ancho
				else texto = "Numero 1";
				break;
			case 1:		
				 // Distancia euclidiana. Si la distancia entre el dedo indice y el gordito supera 90 píxeles.
				if (sqrt(pow(SEvector[0].first.x - SEvector[0].second.x, 2) + pow(SEvector[0].first.y - SEvector[0].second.y, 2)) > 90) texto = "Pistola";
				// Si la distancia es menor a 90 píxeles.
				else texto = "Numero 2";
			break;
			case 2: {

				// Distancia euclidiana.
				int zona_A = sqrt(pow(SEvector[0].first.x - SEvector[0].second.x, 2) + pow(SEvector[0].first.y - SEvector[0].second.y, 2));
				int zona_B = sqrt(pow(SEvector[1].first.x - SEvector[1].second.x, 2) + pow(SEvector[1].first.y - SEvector[1].second.y, 2));

				// Si la distancia entre los dedos es lo suficientemente grande.
				if (zona_B > 70 && zona_A > 70) texto = "Cuernos";
				else if (abs(zona_B - zona_A) > 40) texto = "Saludo Vulcano";
				else texto = "Numero 3";
			}
			break;
			case 3:
				// Por defecto decimos que es el número 4 y si alguna distancia entre Start y End es muy grande le decimos texto = ok;
				texto = "Numero 4";
				// Distancia euclidiana.
				for (unsigned i = 0; i < 3; i++)
					if (sqrt(pow(SEvector[i].first.x - SEvector[i].second.x, 2) + pow(SEvector[i].first.y - SEvector[i].second.y, 2)) > 70) texto = "OK!";
			break;
			case 4:
				texto = "Numero 5";
			break;
			}

		// Pintamos un rectángulo en Frame para que se vea que de ahí salió Roi
		// El color se define con escalar en bgr no en rgb.
		rectangle(frame, rect, Scalar(255,0,0));

		// Pone en frame un texto, en la posicion 400,335; con el estilo FONT_HERSHEY_DUPLEX, tamaño del texto 1, color verde, grosor 2
		putText(frame, texto, Point(400,335), FONT_HERSHEY_DUPLEX, 1, Scalar(0,255,0), 2);
		putText(frame, info, Point(340,75), FONT_HERSHEY_DUPLEX, 1, Scalar(0,255,0), 2);

		// Mostramos las imagenes en nuestra ventana
		imshow("Frame", frame);
		imshow("ROI", roi);
		imshow("Foreground Mask", fgMask);

		// Esperamos 40 milisegundos a que pulsemos una tecla.
		int c = waitKey(40);

		// Si pulsamos la tecla 'q' se sale del bucle infinito y el programa acaba.
		if ((char) c =='q') break;
		// Si pulsamos la tecla '1' define el fondo.
		if ((char) c =='1') learningRate = true;
		// Si pulsamos la tecla '0' volvemos a intentar a aprender el fondo.
		if ((char) c =='0') learningRate = false;

		// Limpiamos texto para siempre mostrar un texto adecuado a lo que hacemos no a lo que habíamos hecho.
		texto = "";

		// Limpiamos el vector.
		SEvector.clear();
	}

	// Liberamos recursos
	cap.release();
	destroyAllWindows();
}
