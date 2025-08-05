

#include <iostream>
#include <dirent.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

// returns os path as imaxe dun cartafol definidas en imgExts
void getFileNames(string dirName, vector<string> &imageFnames)
{
  DIR *dir;
  struct dirent *ent;
  int count = 0;
  string imgExt = "jpg";

  vector<string> files;
  if ((dir = opendir (dirName.c_str())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      //evitar ler nomes non desexados
      if(strcmp(ent->d_name,".") == 0 | strcmp(ent->d_name, "..") == 0) { continue; }
      string temp_name = ent->d_name;
      files.push_back(temp_name);
    }

    // Ordeamos
    std::sort(files.begin(),files.end());
    for(int it=0;it<files.size();it++) {
      string path = dirName;
      string fname = files[it];
      if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos) {
        path.append(fname);
        imageFnames.push_back(path);
      }
    }
    closedir (dir);
  }
}

// Lemos imaxes e etiquetas
void getDataset(string &pathName, int classVal, vector<Mat> &images, vector<int> &labels) {
  vector<string> imageFiles;
  getFileNames(pathName, imageFiles);
  for (int i = 0; i < imageFiles.size(); i++) {
    Mat im = imread(imageFiles[i]);
    images.push_back(im);
    labels.push_back(classVal);
  }
}

// Inicializa HOG
HOGDescriptor hog(
        Size(64, 128), //winSize
        Size(16, 16),  //blocksize
        Size(8, 8),    //blockStride,
        Size(8, 8),    //cellSize,
                9,     //nbins,
                0,     //derivAperture,
                -1,    //winSigma,
                HOGDescriptor::HistogramNormType (0),     //histogramNormType,
                0.2,   //L2HysThresh,
                1,     //gammal correction,
                64,    //nlevels=64
                0);    //signedGradient

// acha HOG
void computeHOG(vector<vector<float> > &hogFeatures, vector<Mat> &images)
{
  for(int y = 0; y < images.size(); y++)
  {
    vector<float> descriptor;
    hog.compute(images[y], descriptor);
    hogFeatures.push_back(descriptor);
  }
}

// formato para SVM
void prepareData(vector<vector<float> > &hogFeatures, Mat &data)
{
  int descriptorSize = hogFeatures[0].size();

  for(int i = 0; i < hogFeatures.size(); i++)
    for(int j = 0; j < descriptorSize; j++)
      data.at<float>(i,j) = hogFeatures[i][j];
}

// Iniciamos SVM
Ptr<SVM> svmInit(float C, float gamma)
{
  Ptr<SVM> svm = SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(SVM::LINEAR);
  svm->setType(SVM::C_SVC);
  return svm;
}

// Adestramos SVM
void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels)
{
  Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
  svm->train(td);
}

// predecimos labels
void svmPredict(Ptr<SVM> svm, Mat &testMat, Mat &testResponse)
{
  svm->predict(testMat, testResponse);
}

// avaliamos o modelo
void svmEvaluate(Mat &testResponse, vector<int> &testLabels, int &correct, float &error)
{
  for(int i = 0; i < testResponse.rows; i++)
  {
    if(testResponse.at<float>(i,0) == testLabels[i])
      correct = correct + 1;
  }
  error = (testLabels.size() - correct)*100.0/testLabels.size();
}

int main()
{
  // Flags para controlar o comportamento do programa
  bool trainModel = 1;
  bool testModel = 1;

  // Path ao dataset do INRIA
  string rootDir = "../../data/images/INRIAPerson/";

  // Paths aos directorios de adestramento e test
  string trainDir = rootDir + "train_64x128_H96/";
  string testDir = rootDir + "test_64x128_H96/";

  // ================================ Adestramento =============================================
  if (trainModel == 1) {
    string trainPosDir = trainDir + "posPatches/";
    string trainNegDir = trainDir + "negPatches/";

    vector<Mat> trainPosImages, trainNegImages;
    vector<int> trainPosLabels, trainNegLabels;

    //  1 positivos e -1 para imaxes negativas
    getDataset(trainPosDir, 1, trainPosImages, trainPosLabels);
    getDataset(trainNegDir, -1, trainNegImages, trainNegLabels);


    cout << "positivas - " << trainPosImages.size() << " , " << trainPosLabels.size() << endl;
    cout << "negativas - " << trainNegImages.size() << " , " << trainNegLabels.size() << endl;


    vector<Mat> trainImages;
    vector<int> trainLabels;
    trainImages = trainPosImages;
    trainImages.insert(trainImages.end(), trainNegImages.begin(), trainNegImages.end());

    trainLabels = trainPosLabels;
    trainLabels.insert(trainLabels.end(), trainNegLabels.begin(), trainNegLabels.end());

    // Acha HOG
    vector<vector<float> > hogTrain;
    computeHOG(hogTrain, trainImages);

    // formato SVM
    int descriptorSize = hogTrain[0].size();
    cout << "Descriptor Size : " << descriptorSize << endl;
    Mat trainData(hogTrain.size(), descriptorSize, CV_32FC1);
    prepareData(hogTrain, trainData);

    // Inicializa SVM
    float C = 0.01, gamma = 0;
    Ptr<SVM> svm = svmInit(C, gamma);
    svmTrain(svm, trainData, trainLabels);
    svm->save("../results/pedestrian.yml");
  }

  // ================================ Test =============================================
  if (testModel == 1) {
    // cargamos o modelo dende disco
    Ptr<SVM> svm = ml::SVM::load("../results/pedestrian.yml");

    string testPosDir = testDir + "posPatches/";
    string testNegDir = testDir + "negPatches/";

    vector<Mat> testPosImages, testNegImages;
    vector<int> testPosLabels, testNegLabels;
    getDataset(testPosDir, 1, testPosImages, testPosLabels);
    getDataset(testNegDir, -1, testNegImages, testNegLabels);

    cout << "positivos - " << testPosImages.size() << " , " << testPosLabels.size() << endl;
    cout << "negativos - " << testNegImages.size() << " , " << testNegLabels.size() << endl;

    // =========== Test sobre imaxes positivas ===============
    // Computa HOG
    vector<vector<float> > hogPosTest;
    computeHOG(hogPosTest, testPosImages);

    // formato para SVM
    int descriptorSize = hogPosTest[0].size();
    cout << "Tamanho do descriptor : " << descriptorSize << endl;
    Mat testPosData(hogPosTest.size(), descriptorSize, CV_32FC1);
    prepareData(hogPosTest, testPosData);
    cout << testPosData.rows << " " << testPosData.cols << endl;

    // clasificamos
    Mat testPosPredict;
    svmPredict(svm, testPosData, testPosPredict);
    int posCorrect = 0;
    float posError = 0;
    svmEvaluate(testPosPredict, testPosLabels, posCorrect, posError);

    // achamos TP FP
    int tp = posCorrect;
    int fp = testPosLabels.size() - posCorrect;
    cout << "TP: " << tp << " FP: " << fp << " total: " << testPosLabels.size() << " error: " << posError << endl;

    // =========== Test sobre imaxe negativas ===============
    // HOG
    vector<vector<float> > hogNegTest;
    computeHOG(hogNegTest, testNegImages);

    // formato para SVM
    cout << "Tamanho do descriptor : " << descriptorSize << endl;
    Mat testNegData(hogNegTest.size(), descriptorSize, CV_32FC1);
    prepareData(hogNegTest, testNegData);

    // calsificamos
    Mat testNegPredict;
    svmPredict(svm, testNegData, testNegPredict);
    int negCorrect = 0;
    float negError = 0;
    svmEvaluate(testNegPredict, testNegLabels, negCorrect, negError);

    // achamos TP e FP
    int tn = negCorrect;
    int fn = testNegLabels.size() - negCorrect;
    cout << "TN: " << tn << " FN: " << fn << " total: " << testNegLabels.size() << " error: " << negError << endl;

    // achamos Precision e Recall
    float precision = tp * 100.0 / (tp + fp);
    float recall = tp * 100.0 / (tp + fn);
    cout << "Precision: " << precision << " Recall: " << recall << endl;
  }
  return 0;
}
