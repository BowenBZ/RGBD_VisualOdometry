/*
 * Sample code to run the RGBD VO system
 */
#include <fstream>
#include <iostream>
#include <boost/timer.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Core>
#include "myslam/config.h"
#include "myslam/frontend.h"
#include "myslam/viewer.h"
#include "myslam/mapmanager.h"
#include "myslam/backend.h"
#include "myslam/frame.h"

void writePosetoFile(ofstream& outputFile, const string& timestamp, const SE3& pose) {
    Vector3d translation = pose.inverse().translation();
    Eigen::Quaterniond rotation = Eigen::Quaterniond(pose.rotationMatrix());
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

    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
    cout<<"Path of dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }
    fin.close();

    cout << "Initializing VO system ..." << endl;

    myslam::Camera::Ptr camera ( new myslam::Camera );
    myslam::FrontEnd::Ptr frontend ( new myslam::FrontEnd );

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

    cout << "Finish initialization!" << endl;

    cout<<"Total "<<rgb_files.size() <<" images from dataset\n\n";
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        Mat color = cv::imread ( rgb_files[i] );
        Mat depth = cv::imread ( depth_files[i], -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            break;
        myslam::Frame::Ptr pFrame = myslam::Frame::CreateFrame(
            rgb_times[i],
            camera,
            color,
            depth);

        cout << "Image #" << i << endl;
        boost::timer timer;
        frontend->AddFrame ( pFrame );
        cout<<"Time cost (s): "<<timer.elapsed()<<endl<<endl;

        if ( frontend->GetState() == myslam::FrontEnd::LOST ) {
            cout << "VO lost" << endl;
            break;        
        }
    }

    cout << "Finished. \nPress <enter> to continue\n"; 
    cin.get();

    ofstream fout (myslam::Config::get<string> ( "output_file" ));
    fout << "# estimated trajectory format" << endl;
    fout << "# timestamp tx ty tz qx qy qz qw" << endl;
    for(auto keyFrameMap: myslam::MapManager::GetInstance().GetAllKeyframes()) {
        auto keyFrame = keyFrameMap.second;
        writePosetoFile(fout, std::to_string(keyFrame->timestamp_), keyFrame->GetPose());
    }
    fout.close();

    if (myslam::Config::get<int> ( "enable_local_optimization" )) {
        backend->Stop();
    }

    if (myslam::Config::get<int> ( "enable_viewer" )) {
        viewer->Close();
    }

    return 0;
}
