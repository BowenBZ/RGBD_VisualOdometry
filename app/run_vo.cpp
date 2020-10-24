// -------------- test the visual odometry -------------
#include <fstream>
#include <iostream>
#include <boost/timer.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "myslam/config.h"
#include "myslam/frontend.h"
#include "myslam/viewer.h"
#include "myslam/map.h"
#include "myslam/backend.h"

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    myslam::Config::setParameterFile ( argv[1] );

    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
    cout<<"dataset: "<<dataset_dir<<endl;
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

    myslam::Camera::Ptr camera ( new myslam::Camera );
    myslam::FrontEnd::Ptr frontend ( new myslam::FrontEnd );
    myslam::Viewer::Ptr viewer (new myslam::Viewer );
    myslam::Map::Ptr map (new myslam::Map );

    frontend->SetMap(map);
    frontend->SetViewer(viewer);
    viewer->SetMap(map);

    myslam::Backend::Ptr backend;
    if (myslam::Config::get<int> ( "enable_local_backend" )) {
        cout << "Enable local backend" << endl;
        backend = myslam::Backend::Ptr(new myslam::Backend);
        backend->SetMap(map);
        backend->SetCamera(camera);
        frontend->SetBackend(backend); 
    }

    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        Mat color = cv::imread ( rgb_files[i] );
        Mat depth = cv::imread ( depth_files[i], -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            break;
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        frontend->addFrame ( pFrame );
        cout<<"VO costs time: "<<timer.elapsed()<<endl<<endl;
        
        if ( frontend->getState() == myslam::FrontEnd::LOST )
            break;        
    }

    if (myslam::Config::get<int> ( "enable_local_backend" )) {
        backend->Stop();
    }

    cout << "Finished. \nPress <enter> to continue\n"; 
    cin.get();

    viewer->Close();

    return 0;
}
