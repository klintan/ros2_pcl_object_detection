/*
Copyright (c) 2020 Andreas Klintberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>
#include "kalman_filter_wrapper.hpp"

KalmanFilterWrapper::KalmanFilterWrapper(std::vector<cv::Vec2f> p)
{
    kalman = new KalmanFilter(4, 4, 0); // 4 measurement and state parameters
    kalman->transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

    // Initialization
    prev_result = p;
    kalman->statePre.at<float>(0) = p[0][0]; // r1
    kalman->statePre.at<float>(1) = p[0][1]; // theta1
    kalman->statePre.at<float>(2) = p[1][0]; // r2
    kalman->statePre.at<float>(3) = p[1][1]; // theta2

    kalman->statePost.at<float>(0) = p[0][0];
    kalman->statePost.at<float>(1) = p[0][1];
    kalman->statePost.at<float>(2) = p[1][0];
    kalman->statePost.at<float>(3) = p[1][1];

    setIdentity(kalman->measurementMatrix);
    setIdentity(kalman->processNoiseCov, Scalar::all(1e-4));
    setIdentity(kalman->measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kalman->errorCovPost, Scalar::all(5));
}

KalmanFilterWrapper::~KalmanFilterWrapper()
{
    delete kalman;
}

// Prediction
std::vector<cv::Vec2f> KalmanFilterWrapper::predict()
{
    Mat prediction = kalman->predict(); // predict the state of the next frame
    prev_result[0][0] = prediction.at<float>(0);
    prev_result[0][1] = prediction.at<float>(1);
    prev_result[1][0] = prediction.at<float>(2);
    prev_result[1][1] = prediction.at<float>(3);
    return prev_result;
}

// Correct the prediction based on the measurement
std::vector<cv::Vec2f> KalmanFilterWrapper::update(vector<Vec2f> measure)
{

    Mat_<float> measurement(4, 1);
    measurement.setTo(Scalar(0));

    measurement.at<float>(0) = measure[0][0];
    measurement.at<float>(1) = measure[0][1];
    measurement.at<float>(2) = measure[1][0];
    measurement.at<float>(3) = measure[1][1];

    Mat estimated = kalman->correct(measurement); // Correct the state of the next frame after obtaining the measurements

    // Update the measurement
    if (estimated.at<float>(0) < estimated.at<float>(2))
    {
        measure[0][0] = estimated.at<float>(0);
        measure[0][1] = estimated.at<float>(1);
        measure[1][0] = estimated.at<float>(2);
        measure[1][1] = estimated.at<float>(3);
    }
    else
    {
        measure[0][0] = estimated.at<float>(2);
        measure[0][1] = estimated.at<float>(3);
        measure[1][0] = estimated.at<float>(0);
        measure[1][1] = estimated.at<float>(1);
    }

    waitKey(1);

    return measure; // return the measurement
}