/*
 * Represents a camara model holding the intrinsics
 * Copyright (C) 2016  Bowen Zhang zbw14@outlook.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef CAMERA_H
#define CAMERA_H

#include "myslam/common_include.h"

namespace myslam
{

// Pinhole RGBD camera model
class Camera
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Camera> Ptr;

    Camera();

    float GetFx() const {
        return fx_;
    }
    float GetFy() const {
        return fy_;
    }
    float GetDepthScale() const {
        return depthScale_;
    }

    Mat GetCameraMatrix() const {
        return cameraMatrix_;
    }

    // coordinate transform: world, camera, pixel
    Vector3d World2Camera( const Vector3d& p_w, const SE3& T_c_w );
    Vector3d Camera2World( const Vector3d& p_c, const SE3& T_c_w );
    Vector2d Camera2Pixel( const Vector3d& p_c );
    Vector3d Pixel2Camera( const Vector2d& p_p, double depth=1 ); 
    Vector3d Pixel2World ( const Vector2d& p_p, const SE3& T_c_w, double depth=1 );
    Vector2d World2Pixel ( const Vector3d& p_w, const SE3& T_c_w );

    // overload functions
    Vector3d Pixel2World ( const KeyPoint& p_p, const SE3& T_c_w, double depth=1 );
    Vector3d Pixel2Camera( const Point2f& p_p, double depth=1 ); 

private:
    float   fx_, fy_, cx_, cy_, depthScale_;  // Camera intrinsics 
    Mat     cameraMatrix_;
};

}
#endif // CAMERA_H
