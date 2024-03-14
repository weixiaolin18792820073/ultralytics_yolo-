#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>

void Detector(YOLO_V8*& p) {
    // std::filesystem::path current_path = std::filesystem::current_path();
    // std::cout << "CreateSession-1" << current_path << std::endl;
    // std::cout << "is_exist = " << std::filesystem::exists(current_path / "images") << std::endl;
    // std::string current_path_str = current_path.string();
    // std::cout << "current_path_str = " << current_path_str << std::endl;
    // std::string imgs_path_str = current_path_str + "/images";
    // std::filesystem::path imgs_path(imgs_path_str);
    // std::cout << "new_path = " << imgs_path[0] << std::endl;
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images";
    // std::cout << "CreateSession-2--" << std::endl;
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        std::cout << "i.path() = " << i.path() << std::endl;
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();

             // 获取文件名
            std::string fileName = i.path().filename().string();
            std::filesystem::path current_path = std::filesystem::current_path();
            std::string img_path_save_str = (current_path / ("result_image/" + fileName)).string();
            // std::string img_path_save_str = (i.path().extension() / "_detect_result.jpg").string();
            // std::cout << "img_path = " << img_path << std::endl;
            // std::cout << img_path << std::endl;
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);
            // std::cout << "save result: " << img_path << std::endl;
            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                // std::cout << "re: " << re << std::endl;
                cv::rectangle(img, re.box, color, 3);
                std::cout << "box.x: " << re.box.x << " box.y: "<< re.box.y << std::endl;

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );
                

            }
            // std::cout << "Press any key to exit" << std::endl;
            // cv::imshow("Result of Detection", img);
            std::cout << "img_path_save_str: " << img_path_save_str << std::endl;
            cv::imwrite(img_path_save_str, img);
            // cv::waitKey(0);
            // cv::destroyAllWindows();
        }
    }
}


void Classifier(YOLO_V8*& p)
{
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path;// / "images"
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png")
        {
            std::string img_path = i.path().string();
            //std::cout << img_path << std::endl;
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            char* ret = p->RunSession(img, res);

            float positionY = 50;
            for (int i = 0; i < res.size(); i++)
            {
                int r = dis(gen);
                int g = dis(gen);
                int b = dis(gen);
                cv::putText(img, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                cv::putText(img, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                positionY += 50;
            }

            cv::imshow("TEST_CLS", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
            //cv::imwrite("E:\\output\\" + std::to_string(k) + ".png", img);
        }

    }
}



int ReadCocoYaml(YOLO_V8*& p) {
    // Open the YAML file
    std::ifstream file("coco.yaml");
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }

    // Extract the names
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':'); // Extract the number before the delimiter
        std::getline(ss, name); // Extract the string after the delimiter
        names.push_back(name);
    }

    p->classes = names;
    return 0;
}


void DetectTest()
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = "yolov8n.onnx";
    params.imgSize = { 640, 640 };
#ifdef USE_CUDA
    params.cudaEnable = true;
    std::cout << params.cudaEnable << std::endl;
    std::cout << "yolo_detect_v8" << std::endl;
    // GPU FP32 inference
    params.modelType = YOLO_DETECT_V8;
    // GPU FP16 inference
    //Note: change fp16 onnx model
    //params.modelType = YOLO_DETECT_V8_HALF;
    std::cout <<YOLO_DETECT_V8 << std::endl;
#else
    // CPU inference
    params.modelType = YOLO_DETECT_V8;
    std::cout << "yolo_detect_v8=-2" << std::endl;
    params.cudaEnable = false;
    std::cout << params.modelType << std::endl;
#endif
    std::cout << "yolo_detect_v8=-3" << std::endl;
    yoloDetector->CreateSession(params);
    std::cout << params.modelPath << std::endl;
    std::cout << "CreateSession" << std::endl;
    Detector(yoloDetector);
}


void ClsTest()
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    std::string model_path = "yolov8n.onnx";
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params{ model_path, YOLO_CLS, {224, 224} };
    yoloDetector->CreateSession(params);
    
    Classifier(yoloDetector);
}


int main()
{
    // DetectTest();
    ClsTest();
}
