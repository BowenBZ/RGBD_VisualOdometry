#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace myslam
{
namespace math {

template<typename FLOAT>
class SE3 {
    typedef Eigen::Matrix<FLOAT, 3, 3> Matrix3;
    typedef Eigen::Matrix<FLOAT, 4, 4> Matrix4;
    typedef Eigen::Matrix<FLOAT, 3, 4> Matrix34;
    typedef Eigen::Matrix<FLOAT, 3, 1> Vector3;
    typedef Eigen::Matrix<FLOAT, 6, 1> Vector6;

public:

#pragma constructor

    SE3() {
        rotation_ = Matrix3::Identity();
        position_ = Vector3::Zero();
    }

    SE3(const Matrix3& rotation, const Vector3& position): rotation_(rotation), position_(position) {
        assert(SE3::isValidRotation(rotation));
    }

    SE3(const Matrix4& transform) {
        assert(SE3::isValidTransform(transform));
        rotation_ = transform.template block<3, 3>(0, 0);
        position_ = transform.template block<3, 1>(0, 3);
    }

    SE3& operator=(const SE3& other) {
        this->rotation_ = other.rotation_;
        this->position_ = other.position_;
        this->algebra_ = other.algebra_;
        return *this;
    }

#pragma getter

    // Return the 3x3 rotation matrix
    Matrix3 rotationMatrix() const {
        return rotation_;
    }

    // Return the translation
    Vector3 translation() const {
        return position_;
    }

    // Return the rotation and translation matrix
    Matrix34 matrix3x4() const {
        Matrix34 output;
        output.template block<3, 3>(0, 0) = rotation_;
        output.template block<3, 1>(0, 3) = position_;
        return output;
    }

    // Return the 4x4 matrix
    Matrix4 matrix() const {
        Matrix4 output = Matrix4::Identity();
        output.template block<3, 3>(0, 0) = rotation_;
        output.template block<3, 1>(0, 3) = position_;
        return output;
    }

    // Return the Lie algebra
    Vector6 log() {
        if (!algebraCalculated_) {
            algebra_ = SE3::log(rotation_, position_);
            algebraCalculated_ = true;
        }
        return algebra_;
    }

    // Return the inverse
    SE3 inverse() const {
        Matrix3 inverseR = this->rotation_.transpose();
        Vector3 inverseP = -1.0 * inverseR * this->position_;
        return SE3(inverseR, inverseP);
    }

#pragma exp & log

    // Return the Lie group
    static SE3 exp(const Vector6& algebra) {
        // Calculate so3 -> SO3
        Vector3 phi = algebra.template tail<3>();
        FLOAT theta = phi.norm();
        Vector3 a = phi.normalized();
        Matrix3 aUp;
        aUp << 0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0;

        FLOAT cosTheta = std::cos(theta);
        FLOAT sinTheta = std::sin(theta);
        Matrix3 R = cosTheta * Matrix3::Identity() + (1 - cosTheta) * a * a.transpose() + sinTheta * aUp;

        // Calculate translation part
        Matrix3 J = sinTheta / theta * Matrix3::Identity() + (1 - sinTheta / theta) * a * a.transpose() + (1 - cosTheta) / theta * aUp;
        Vector3 rho = algebra.template head<3>();
        Vector3 p = J * rho;

        return SE3(R, p);
    }

    // Return the Lie algebra
    static Vector6 log(const Matrix3& rotation, const Vector3& position) {
        assert(SE3::isValidRotation(rotation));

        Vector6 algebra;

        // SO3 -> so3
        FLOAT theta = std::acos((rotation.trace() - 1.0) * 0.5);

        bool solveSuccess = false;
        Vector3 a;
        Eigen::EigenSolver<Matrix3> solver(rotation);
        for (size_t i = 0; i < 3; ++i) {
            if (std::abs(solver.eigenvalues()[i].real() - 1.0) < 1e-6) {
                solveSuccess = true;
                a = solver.eigenvectors().col(i).real();
                break;
            }
        }
        assert(solveSuccess);

        Vector3 phi = theta * a;
        algebra.template tail<3>() = phi;

        // Calculate translation part
        Matrix3 aUp;
        aUp << 0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0;
        FLOAT cosTheta = std::cos(theta);
        FLOAT sinTheta = std::sin(theta);
        Matrix3 J = sinTheta / theta * Matrix3::Identity() + (1 - sinTheta / theta) * a * a.transpose() + (1 - cosTheta) / theta * aUp;

        Vector3 rho = J.inverse() * position;
        algebra.template head<3>() = rho;

        return algebra;
    }

#pragma operation

    SE3 operator*(const SE3& other) const {
        Matrix4 result = this->matrix() * other.matrix();
        return SE3(result);
    }

    Vector3 operator*(const Vector3& point) const {
        return this->rotation_ * point + this->position_;
    }

private:
    Matrix3 rotation_;
    Vector3 position_;
    
    bool algebraCalculated_;
    // [rho, phi]^T, phi is so3. Only get initialized when log is called the first time.
    Vector6 algebra_;

    static bool isValidRotation(const Matrix3& rotation) {
        if (!(rotation.transpose() * rotation).isApprox(Matrix3::Identity(), 1e-6)) {
            return false;
        }

        if (std::abs(rotation.determinant() - 1.0) > 1e-6) {
            return false;
        }

        return true;
    }

    static bool isValidTransform(const Matrix4& transform) {
        if (transform(3, 0) != 0.0 || transform(3, 1) != 0.0 ||
            transform(3, 2) != 0.0 || transform(3, 3) != 1.0) {
                return false;
            }

        return isValidRotation(transform.template block<3, 3>(0, 0));
    }
};

}
}