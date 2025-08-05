#include <iostream>
#include <dirent.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include "cropEyeRegion.h"

using namespace cv::ml;
using namespace cv;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 1


void getFileNames(string dirName, vector<string> &imageFnames)
{
  DIR *dir;
  struct dirent *ent;
  int count = 0;

  //image extensions to be found
  string imgExt = "jpg";

  vector<string> files;

  if ((dir = opendir (dirName.c_str())) != NULL)
  {
    while ((ent = readdir (dir)) != NULL)
    {
      if(strcmp(ent->d_name,".") == 0 | strcmp(ent->d_name, "..") == 0)
      {
        continue;
      }
      string temp_name = ent->d_name;
      files.push_back(temp_name);
    }

    // ordenamos nomes
    std::sort(files.begin(),files.end());
    for(int it=0;it<files.size();it++)
    {
      string path = dirName;
      string fname = files[it];

      if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
    }
    closedir (dir);
  }
}

void loadTrainTestLabel(string &pathName, vector<Mat> &trainImages, vector<Mat> &testImages, vector<int> &trainLabels, vector<int> &testLabels, int classVal, float testFraction = 0.2)
{
  vector<string> imageFiles;

  getFileNames(pathName, imageFiles);

  int totalImages = imageFiles.size();
  int nTest = (int)(testFraction*totalImages);

  for (int counter = 0; counter < totalImages ; counter++)
  {
    Mat image = imread(imageFiles[counter]);
    if(counter < nTest)
    {
        testImages.push_back(image);
        testLabels.push_back(classVal);
    }
    else
    {
        trainImages.push_back(image);
        trainLabels.push_back(classVal);
    }
  }
}

HOGDescriptor hog(
        Size(96,32), //winSize
        Size(8,8), //blocksize
        Size(8,8), //blockStride,
        Size(4,4), //cellSize,
                9, //nbins,
                0, //derivAper,
                4, //winSigma,
                HOGDescriptor::HistogramNormType (1), //histogramNormType,
                0.2, //L2HysThresh,
                1,//gammal correction,
                64,//nlevels=64
                1);//signedGradient

void CreateHOG(vector<vector<float> > &HOG, vector<Mat> &Images)
{
  for(int y = 0; y < Images.size(); y++)
  {
    vector<float> descriptors;
    hog.compute(Images[y],descriptors);
    HOG.push_back(descriptors);
  }
}

void ConvertVectortoMatrix(vector<vector<float> > &HOG, Mat &samples)
{
  int descriptor_size = HOG[0].size();

  for(int i = 0; i < HOG.size(); i++)
    for(int j = 0; j < descriptor_size; j++)
      samples.at<float>(i,j) = HOG[i][j];
}

void getSVMParams(Ptr<SVM> svm)
{
  cout << "Kernel type     : " << svm->getKernelType() << endl;
  cout << "Type            : " << svm->getType() << endl;
  cout << "C               : " << svm->getC() << endl;
  cout << "Degree          : " << svm->getDegree() << endl;
  cout << "Nu              : " << svm->getNu() << endl;
  cout << "Gamma           : " << svm->getGamma() << endl;
}

Ptr<SVM> svmInit(float C, float gamma)
{
  Ptr<SVM> svm = SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(SVM::RBF);
  svm->setType(SVM::C_SVC);

  return svm;
}

void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels)
{
  Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
  svm->train(td);
  svm->save("../results/eyeGlassClassifierModel.yml");
}

void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat )
{
  svm->predict(testMat, testResponse);
}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels)
{
  for(int i = 0; i < testResponse.rows; i++)
  {
    // cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
    if(testResponse.at<float>(i,0) == testLabels[i])
      count = count + 1;
  }
  accuracy = (count/testResponse.rows)*100;
}

int main(int argc, char **argv)
{
  vector<Mat> trainImages;
  vector<Mat> testImages;
  vector<int> trainLabels;
  vector<int> testLabels;

  // Path1 e a clase 0 e Path2 a clase 1
  string path1 = "../../data/images/glassesDataset/cropped_withoutGlasses2/";
  string path2 = "../../data/images/glassesDataset/cropped_withGlasses2/";

  // cargamos as imaxes
  loadTrainTestLabel(path1, trainImages, testImages, trainLabels, testLabels, 0);
  loadTrainTestLabel(path2, trainImages, testImages, trainLabels, testLabels, 1);

  ////////// calculo de HOG  ///////////
  std::vector<std::vector<float> > trainHOG;
  std::vector<std::vector<float> > testHOG;
  CreateHOG(trainHOG, trainImages);
  CreateHOG(testHOG, testImages);

  int descriptor_size = trainHOG[0].size();
  cout << "Tamanho do descritor : " << descriptor_size << endl;

  // formato para SVM
  Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
  Mat testMat(testHOG.size(), descriptor_size, CV_32FC1);

  ConvertVectortoMatrix(trainHOG, trainMat);
  ConvertVectortoMatrix(testHOG, testMat);

  float C = 2.5, gamma = 0.02;

  Mat testResponse;
  Ptr<SVM> model = svmInit(C, gamma);

  ///////////  SVM Training  ////////////////
  svmTrain(model, trainMat, trainLabels);

  ///////////  SVM Testing  ////////////////
  svmPredict(model, testResponse, testMat);

  ////////////// achamos Accuracy   ///////////
  float count = 0;
  float accuracy = 0 ;
  getSVMParams(model);
  SVMevaluate(testResponse, count, accuracy, testLabels);

  /////////// Testing sobre unha imaxe   ////////////////////
  string filename = "../../data/images/glassesDataset/glasses_3.jpg";
  if (argc == 2)
  {
    filename = argv[1];
  }

  Mat testImage = imread(filename);

  // rexion dos ollos
  Mat cropped = getCroppedEyeRegion(testImage);

  vector<Mat> testImageArray;
  testImageArray.push_back(cropped);

  // HOG
  std::vector<std::vector<float> > testHOGArray;
  CreateHOG(testHOGArray, testImageArray);

  // convertimos descriptore a Mat
  Mat testSample(testHOGArray.size(), descriptor_size, CV_32FC1);
  ConvertVectortoMatrix(testHOGArray, testSample);

  // Cargamos o modelo dende disco
  Ptr<SVM> savedModel = StatModel::load<SVM>("../results/eyeGlassClassifierModel.yml");
  Mat pred;
  svmPredict(savedModel, pred, testSample);

  cout << "Predicion : " << pred.at<float>(0,0) << endl;
  imshow("Imaxe test", testImage);
  imshow("Rexion de ollos", cropped);
  waitKey(0);
  destroyAllWindows();

  return 0;
}
