/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
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

#include "myslam/camera.h"
#include "myslam/config.h"
#include "myslam/util.h"

namespace myslam
{

Camera::Camera()
{
    fx_ = Config::get<float>("camera.fx");
    fy_ = Config::get<float>("camera.fy");
    cx_ = Config::get<float>("camera.cx");
    cy_ = Config::get<float>("camera.cy");
    depthScale_ = Config::get<float>("camera.depth_scale");

    cameraMatrix_ = ( cv::Mat_<double>(3,3) <<
                    fx_, 0,   cx_,
                    0,   fy_, cy_,
                    0,   0,   1 );
}

Vector3d Camera::World2Camera ( const Vector3d& p_w, const SE3& T_c_w )
{
    return T_c_w * p_w;
}

Vector3d Camera::Camera2World ( const Vector3d& p_c, const SE3& T_c_w )
{
    return T_c_w.inverse() * p_c;
}

Vector2d Camera::Camera2Pixel ( const Vector3d& p_c )
{
    return Vector2d (
        fx_ * p_c ( 0,0 ) / p_c ( 2,0 ) + cx_,
        fy_ * p_c ( 1,0 ) / p_c ( 2,0 ) + cy_
    );
}

Vector3d Camera::Pixel2Camera ( const Vector2d& p_p, double depth )
{
    return Vector3d (
        ( p_p ( 0,0 )-cx_ ) *depth/fx_,
        ( p_p ( 1,0 )-cy_ ) *depth/fy_,
        depth
    );
}

Vector3d Camera::Pixel2Camera ( const Point2f& p_p, double depth )
{
    return Pixel2Camera(toVec2d(p_p), depth);
}

Vector2d Camera::World2Pixel ( const Vector3d& p_w, const SE3& T_c_w )
{
    return Camera2Pixel ( World2Camera ( p_w, T_c_w ) );
}

Vector3d Camera::Pixel2World ( const Vector2d& p_p, const SE3& T_c_w, double depth )
{
    return Camera2World ( Pixel2Camera ( p_p, depth ), T_c_w );
}

Vector3d Camera::Pixel2World ( const KeyPoint& p_p, const SE3& T_c_w, double depth )
{
    return Pixel2World(toVec2d(p_p), T_c_w, depth);
}

}
