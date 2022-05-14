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

#ifndef MYSLAM_G2O_TYPES_H
#define MYSLAM_G2O_TYPES_H

#include "myslam/common_include.h"
#include "myslam/camera.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

namespace myslam
{
// g2o::VertexSE3Expmap uses g2o::SE3Quat as the element
// g2o::SE3Quat first 3 elements are rotation, last 3 elements are translation

/// vertex and edges used in g2o ba
/// Sophus::SE3 with the first 3 of translation, and last 3 of rotation
class VertexPose : public g2o::BaseVertex<6, SE3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override { _estimate = SE3(); }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = SE3::exp(update_eigen) * _estimate;
    }

    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }
};

// edge for pose vertex, not optimize the mappoint
class UnaryEdgeProjection : public g2o::BaseUnaryEdge<2, Vector2d, VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    UnaryEdgeProjection(const Vector3d &pos, const Camera::Ptr& camera) : _mappointPos(pos), _camera(camera) {}

    virtual void computeError() override
    {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        SE3 T = v->estimate();
        _error = _measurement - _camera->Camera2Pixel(T * _mappointPos);
    }

    virtual void linearizeOplus() override
    {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        SE3 T = v->estimate();
        Vector3d pos_cam = T * _mappointPos;
        double fx = _camera->GetFx();
        double fy = _camera->GetFy();
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);
        double Zinv2 = Zinv * Zinv;
        _jacobianOplusXi
        << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2, -fx - fx * X * X * Zinv2, fx * Y * Zinv,
        0, -fy * Zinv, fy * Y / (Z * Z), fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2, -fy * X * Zinv;
        }

        virtual bool read( std::istream& in ) override {}
        virtual bool write(std::ostream& os) const override {}

private:
    Vector3d _mappointPos;
    Camera::Ptr _camera;
};

/// Vertex for mappoint
class VertexMappoint : public g2o::BaseVertex<3, Vector3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual void setToOriginImpl() override { _estimate = Vector3d::Zero(); }

    virtual void oplusImpl(const double *update) override {
        _estimate[0] += update[0];
        _estimate[1] += update[1];
        _estimate[2] += update[2];
    }

    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }
};

// edge for pose vertex and mappoint vertex
class BinaryEdgeProjection : public g2o::BaseBinaryEdge<2, Vector2d, VertexPose, VertexMappoint> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    BinaryEdgeProjection(const Camera::Ptr& camera) : _camera(camera) { }

    virtual void computeError() override {
        const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
        const VertexMappoint *v1 = static_cast<VertexMappoint *>(_vertices[1]);
        SE3 T = v0->estimate();
        _error = _measurement - _camera->Camera2Pixel(T * v1->estimate());
    }

    virtual void linearizeOplus() override {
        const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
        const VertexMappoint *v1 = static_cast<VertexMappoint *>(_vertices[1]);
        SE3 T = v0->estimate();
        Vector3d pos_world = v1->estimate();
        Vector3d pos_cam = T * pos_world;
        double fx = _camera->GetFx();
        double fy = _camera->GetFy();
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);
        double Zinv2 = Zinv * Zinv;
        _jacobianOplusXi 
        << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2, -fx - fx * X * X * Zinv2, fx * Y * Zinv, 
        0, -fy * Zinv, fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2, -fy * X * Zinv;

        _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) * T.rotationMatrix();
    }

    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }

   private:
    Camera::Ptr _camera;
};


} // namespace

#endif // MYSLAM_G2O_TYPES_H
