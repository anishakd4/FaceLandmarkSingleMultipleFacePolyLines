#include<dlib/image_processing.h>
#include<dlib/image_processing/frontal_face_detector.h>
#include<dlib/opencv.h>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace dlib;
using namespace std;

//draw polyline
void drawPolyline(cv::Mat &image, full_object_detection landmarks, int start, int end, bool isClosed=false){
    std::vector<cv::Point> points;
    for(int i=start; i<=end; i++){
        points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
    }
    cv::polylines(image, points, isClosed, cv::Scalar(0, 255, 255), 2, 16);
}

void drawPolylines(cv::Mat &image, full_object_detection landmarks){
    drawPolyline(image, landmarks, 0, 16);              //jaw line
    drawPolyline(image, landmarks, 17, 21);             //left eyebrow
    drawPolyline(image, landmarks, 22, 26);             //right eyebrow
    drawPolyline(image, landmarks, 27, 30);             //Nose bridge
    drawPolyline(image, landmarks, 30, 35, true);       //lower nose
    drawPolyline(image, landmarks, 36, 41, true);       //left eye
    drawPolyline(image, landmarks, 42, 47, true);       //right eye
    drawPolyline(image, landmarks, 48, 59, true);       //outer lip
    drawPolyline(image, landmarks, 60, 67, true);       //inner lip
}

int main(){

    //read images
    cv::Mat imageSingle = cv::imread("../assets/anish.jpg");
    cv::Mat imageMultiple = cv::imread("../assets/anish2.jpg");

    //create images clone to work on
    cv::Mat imageSingleClone = imageSingle.clone();
    cv::Mat imageMultipleClone = imageMultiple.clone();

    //Define face detector
    frontal_face_detector faceDetector = get_frontal_face_detector();

    //define landmark detector
    shape_predictor landmarkDetector;

    //load the face landmark model
    deserialize("../dlibAndModel/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

    //convert opencv image format to dlib image format
    cv_image<bgr_pixel> dlibImageSingle(imageSingleClone);
    cv_image<bgr_pixel> dlibImageMultiple(imageMultipleClone);

    //detect faces in the image
    std::vector<rectangle> facesSingle = faceDetector(dlibImageSingle);
    std::vector<rectangle> facesMultiple = faceDetector(dlibImageMultiple);

    //loop over all detected faces
    for(int i=0; i<facesSingle.size(); i++){
        //for each face run landmark detector
        full_object_detection landmarks = landmarkDetector(dlibImageSingle, facesSingle[i]);

        //print number of landmarks detected
        cout<<"Number of face landmarks detected: "<<landmarks.num_parts()<<endl;

        //draw polyline around face landmarks
        drawPolylines(imageSingleClone, landmarks);
    }
    for(int i=0; i<facesMultiple.size(); i++){
        //for each face run landmark detector
        full_object_detection landmarks = landmarkDetector(dlibImageMultiple, facesMultiple[i]);

        //print number of landmarks detected
        cout<<"Number of face landmarks detected: "<<landmarks.num_parts()<<endl;

        //draw polyline around face landmarks
        drawPolylines(imageMultipleClone, landmarks);
    }

    //create windows to display images
    cv::namedWindow("single face", cv::WINDOW_NORMAL);
    cv::namedWindow("single face landmarks", cv::WINDOW_NORMAL);
    cv::namedWindow("multiple face", cv::WINDOW_NORMAL);
    cv::namedWindow("multiple face landmarks", cv::WINDOW_NORMAL);

    //display images
    cv::imshow("single face", imageSingle);
    cv::imshow("single face landmarks", imageSingleClone);
    cv::imshow("multiple face", imageMultiple);
    cv::imshow("multiple face landmarks", imageMultipleClone);

    //press esc to exit program
    cv::waitKey(0);

    //close all the opened windows
    cv::destroyAllWindows();

    return 0;
}