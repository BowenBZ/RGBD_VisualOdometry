/*
 * Sample code to run the RGBD VO system
 */
#include <fstream>
#include <iostream>
#include <boost/timer/timer.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Core>

#include "myslam/config.h"
#include "myslam/frontend.h"
#include "myslam/viewer.h"
#include "myslam/mapmanager.h"
#include "myslam/backend.h"
#include "myslam/frame.h"

void writePosetoFile(ofstream& outputFile, const string& timestamp, const SE3& Twc) {
    Vector3d translation = Twc.translation();
    Eigen::Quaterniond rotation = Eigen::Quaterniond(Twc.rotationMatrix());
    outputFile << timestamp << ' ' << translation[0] << ' ' << translation[1] << ' ' << translation[2] 
                << ' ' << rotation.coeffs()[0] << ' ' << rotation.coeffs()[1] << ' ' 
                << rotation.coeffs()[2] << ' ' << rotation.coeffs()[3] << endl;
}

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }
    myslam::Config::setParameterFile ( argv[1] );

    string datasetDir = myslam::Config::get<string> ( "dataset_dir" );
    string datasetEntryFile = datasetDir + "/associate.txt";
    cout << "Path of dataset: " << datasetEntryFile << endl;
    ifstream fin ( datasetEntryFile );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }
    vector<string> rgbFiles, depthFiles;
    vector<double> rgbTimes, depthTimes;
    while ( !fin.eof() )
    {
        string rgbTime, rgbFile, depthTime, depthFile;
        fin>>rgbTime>>rgbFile>>depthTime>>depthFile;
        if (rgbTime.size() == 0) {
            break;
        }

        rgbTimes.push_back ( atof ( rgbTime.c_str() ) );
        depthTimes.push_back ( atof ( depthTime.c_str() ) );
        rgbFiles.push_back ( datasetDir+"/"+rgbFile );
        depthFiles.push_back ( datasetDir+"/"+depthFile );

        if ( !fin.good() ) {
            break;
        }
    }
    fin.close();
    cout<< "Total " << rgbFiles.size() << " images from dataset\n\n";

    const string outputPath = myslam::Config::get<string> ( "output_file" );
    ofstream fout (outputPath);
    fout << "# estimated trajectory format" << endl;
    fout << "# timestamp tx ty tz qx qy qz qw" << endl;

    cout << "Initializing VO system ..." << endl;
    myslam::Camera::Ptr camera ( new myslam::Camera );
    myslam::Frontend::Ptr frontend ( new myslam::Frontend );
    myslam::Viewer::Ptr viewer;
    if (myslam::Config::get<int> ( "enable_viewer" )) {
        cout << "Enable to show image" << endl; 
        viewer = myslam::Viewer::Ptr( new myslam::Viewer );
        frontend->SetViewer(viewer);
    }
    myslam::Backend::Ptr backend;
    if (myslam::Config::get<int> ( "enable_local_optimization" )) {
        cout << "Enable local optimization" << endl;
        backend = myslam::Backend::Ptr(new myslam::Backend(camera));
        frontend->SetBackend(backend); 
    }
    cout << "Finish initialization!\n\n" << endl;
    
    bool pauseEveryFrame = (myslam::Config::get<int>("single_step") == 1);
    for ( size_t i = 0; i < rgbFiles.size(); ++i )
    {
        if (pauseEveryFrame) {
            cin.get();
        }
        Mat color = cv::imread ( rgbFiles[i] );
        Mat depth = cv::imread ( depthFiles[i], -1 );
        if ( color.data == nullptr || depth.data == nullptr ) {
            cout << "Frame missing" << endl;
            break;
        }
        myslam::Frame::Ptr pFrame = myslam::Frame::CreateFrame(
            rgbTimes[i],
            camera,
            color,
            depth);

        cout << "Image #" << i << endl;
        boost::timer::cpu_timer timer;

        frontend->AddFrame ( pFrame );
        
        boost::timer::cpu_times elapsed_times(timer.elapsed());
        cout << "Time cost (ms): " << (elapsed_times.user + elapsed_times.system) / pow(10.0, 6.0) << endl << endl;

        if ( frontend->GetState() == myslam::Frontend::LOST ) {
            cout << "VO lost" << endl;
            break;
        }

        writePosetoFile(fout, std::to_string(pFrame->timestamp_), pFrame->GetPose().inverse());
    }

    fout.close();
    cout << "Finished. \nWrote trajectory to " << outputPath << endl; 

    if (myslam::Config::get<int> ( "enable_local_optimization" )) {
        backend->Stop();
    }

    if (myslam::Config::get<int> ( "enable_viewer" )) {
        viewer->Close();
    }

    cout << "\nPress <enter> to continue\n"; 
    cin.get();

    return 0;
}
