/*
 * Sample code to run the RGBD VO system
 */

#include <myslam/myslam.hpp>
 
#include <boost/timer/timer.hpp>

using namespace std;

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
    std::vector<string> rgbFiles, depthFiles;
    std::vector<double> rgbTimes, depthTimes;
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
    myslam::Frontend::Ptr frontend ( new myslam::Frontend(camera) );
    myslam::Viewer::Ptr viewer;
    const bool enable_viewer = myslam::Config::get<int> ( "enable_viewer" );
    if (enable_viewer) {
        cout << "Enable to show image" << endl; 
        viewer = myslam::Viewer::Ptr( new myslam::Viewer );
        frontend->SetViewer(viewer);
    }
    cout << "Finish initialization!\n\n" << endl;
    
    bool pauseEveryFrame = (myslam::Config::get<int>("single_step") == 1);
    for ( size_t i = 0; i < rgbFiles.size(); ++i )
    {
        if (pauseEveryFrame) {
            cin.get();
        }
        cv::Mat color = cv::imread ( rgbFiles[i] );
        cv::Mat depth = cv::imread ( depthFiles[i], -1 );
        if ( color.data == nullptr || depth.data == nullptr ) {
            cout << "Frame missing" << endl;
            break;
        }

        myslam::Measurement measurement {
            rgbTimes[i],
            color,
            depth
        };

        cout << "Image #" << i << endl;
        boost::timer::cpu_timer timer;

        frontend->AddFrame(measurement);
        
        boost::timer::cpu_times elapsed_times(timer.elapsed());
        cout << "Time cost (ms): " << (elapsed_times.user + elapsed_times.system) / pow(10.0, 6.0) << endl << endl;

        if ( frontend->GetState() == myslam::Frontend::LOST ) {
            cout << "VO lost" << endl;
            break;
        }

        SE3 T_c_w = frontend->GetPose();
        writePosetoFile(fout, std::to_string(rgbTimes[i]), T_c_w.inverse());

        if (enable_viewer) {
            viewer->SingleStep();
        }
    }

    frontend->Stop();

    fout.close();
    cout << "Finished. \nWrote trajectory to " << outputPath << endl; 
    cout << "\nPress <enter> to continue\n"; 
    cin.get();

    return 0;
}
