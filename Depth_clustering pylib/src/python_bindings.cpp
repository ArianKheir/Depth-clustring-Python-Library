/*
 * Python bindings for depth_clustering library using pybind11
 * 
 * This file provides comprehensive Python bindings for all classes and functions
 * in the depth_clustering library.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/matrix.h>  
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/stl_bind.h>

// Core includes
#include "utils/rich_point.h"
#include "utils/cloud.h"
#include "utils/pose.h"
#include "utils/radians.h"
#include "utils/bbox.h"
#include "utils/folder_reader.h"
#include "utils/velodyne_utils.h"
#include "utils/useful_typedefs.h"
#include "utils/mem_utils.h"
#include "utils/timer.h"

// Projection includes
#include "projections/projection_params.h"
#include "projections/cloud_projection.h"
#include "projections/ring_projection.h"
#include "projections/spherical_projection.h"

// Clustering includes
#include "clusterers/abstract_clusterer.h"
#include "clusterers/euclidean_clusterer.h"
#include "clusterers/image_based_clusterer.h"

// Ground removal includes
#include "ground_removal/depth_ground_remover.h"

// Communication includes
#include "communication/identifiable.h"
#include "communication/abstract_client.h"
#include "communication/abstract_sender.h"

// Visualization includes
#include "visualization/cloud_saver.h"

// Image labeling includes
#include "image_labelers/abstract_image_labeler.h"
#include "image_labelers/linear_image_labeler.h"
#include "image_labelers/dijkstra_image_labeler.h"
#include "image_labelers/pixel_coords.h"
#include "image_labelers/hash_queue.h"
#include "image_labelers/diff_helpers/diff_factory.h"
#include "image_labelers/diff_helpers/abstract_diff.h"
#include "image_labelers/diff_helpers/angle_diff.h"
#include "image_labelers/diff_helpers/line_dist_diff.h"
#include "image_labelers/diff_helpers/simple_diff.h"


// OpenCV includes
#include <opencv2/opencv.hpp>

// PCL includes (if available)
#ifdef PCL_FOUND
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#endif


// for faster np to point cloude 
#include <cmath>
#include <vector>
#include <limits>
#include <memory> 

namespace py = pybind11;
using namespace depth_clustering;
using Cloud = depth_clustering::Cloud;
using ClusterMap = std::unordered_map<uint16_t, Cloud>;
using DepthMap = std::unordered_map<uint16_t, cv::Mat>; 
namespace pybind11 { namespace detail {

// Type caster for cv::Mat
template <> struct type_caster<cv::Mat> {
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    // Python -> C++
    bool load(handle src, bool) {
        if (!isinstance<array>(src)) {
            return false;
        }
        array arr = reinterpret_borrow<array>(src);
        
        // Get array properties
        buffer_info info = arr.request();
        
        int ndims = info.ndim;
        if (ndims > 3) {
            return false;
        }
        
        // Determine OpenCV type
        int cv_type;
        if (info.format == format_descriptor<uint8_t>::format()) {
            cv_type = CV_8U;
        } else if (info.format == format_descriptor<int8_t>::format()) {
            cv_type = CV_8S;
        } else if (info.format == format_descriptor<uint16_t>::format()) {
            cv_type = CV_16U;
        } else if (info.format == format_descriptor<int16_t>::format()) {
            cv_type = CV_16S;
        } else if (info.format == format_descriptor<int32_t>::format()) {
            cv_type = CV_32S;
        } else if (info.format == format_descriptor<float>::format()) {
            cv_type = CV_32F;
        } else if (info.format == format_descriptor<double>::format()) {
            cv_type = CV_64F;
        } else {
            return false;
        }
        
        // Handle different dimensions
        if (ndims == 2) {
            value = cv::Mat(info.shape[0], info.shape[1], cv_type, info.ptr);
        } else if (ndims == 3) {
            value = cv::Mat(info.shape[0], info.shape[1], CV_MAKETYPE(cv_type, info.shape[2]), info.ptr);
        } else {
            return false;
        }
        
        return true;
    }

    // C++ -> Python
    static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
        std::string format = format_descriptor<uint8_t>::format();
        size_t elemsize = sizeof(uint8_t);
        int depth = m.depth();
        
        switch(depth) {
            case CV_8U:
                format = format_descriptor<uint8_t>::format();
                elemsize = sizeof(uint8_t);
                break;
            case CV_8S:
                format = format_descriptor<int8_t>::format();
                elemsize = sizeof(int8_t);
                break;
            case CV_16U:
                format = format_descriptor<uint16_t>::format();
                elemsize = sizeof(uint16_t);
                break;
            case CV_16S:
                format = format_descriptor<int16_t>::format();
                elemsize = sizeof(int16_t);
                break;
            case CV_32S:
                format = format_descriptor<int32_t>::format();
                elemsize = sizeof(int32_t);
                break;
            case CV_32F:
                format = format_descriptor<float>::format();
                elemsize = sizeof(float);
                break;
            case CV_64F:
                format = format_descriptor<double>::format();
                elemsize = sizeof(double);
                break;
            default:
                throw std::runtime_error("Unsupported cv::Mat depth");
        }
        
        std::vector<size_t> shape = {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols)};
        std::vector<size_t> strides = {static_cast<size_t>(m.step[0]), static_cast<size_t>(m.step[1])};
        
        if (m.channels() > 1) {
            shape.push_back(m.channels());
            strides.push_back(elemsize);
        }
        
        // Create numpy array with the Mat data (no copy, shares memory)
        array a(pybind11::dtype(format), shape, strides, m.data);
        
        return a.release();
    }
};

}}

class PyCloudProjection : public CloudProjection {
public:
    // Inherit the constructors
    using CloudProjection::CloudProjection;

    // Trampoline for InitFromPoints, with the CORRECT signature
    void InitFromPoints(const RichPoint::AlignedVector& points) override {
        PYBIND11_OVERRIDE_PURE(
            void,                          /* Return type */
            CloudProjection,               /* Parent class */
            InitFromPoints,                /* Name of function */
            points                         /* Arguments */
        );
    }

    // Trampoline for Clone, which was missing before
    CloudProjection::Ptr Clone() const override {
        PYBIND11_OVERRIDE_PURE(
            CloudProjection::Ptr,          /* Return type */
            CloudProjection,               /* Parent class */
            Clone                          /* Name of function */
                                           /* No arguments */
        );
    }
};

template <typename T>
class PyClient : public AbstractClient<T> {
public:
    using AbstractClient<T>::AbstractClient; // Inherit constructors

    void OnNewObjectReceived(const T& object, const int sender_id) override {
        PYBIND11_OVERRIDE_PURE(
            void,                   /* Return type */
            AbstractClient<T>,      /* Parent class */
            OnNewObjectReceived,    /* Function name */
            object,                 /* Argument 1 */
            sender_id               /* Argument 2 */
        );
    }
};
template <>
class PyClient<DepthMap> : public AbstractClient<DepthMap> {
public:
    using AbstractClient<DepthMap>::AbstractClient; // Inherit constructor
    void OnNewObjectReceived(const DepthMap& obj, const int id) override {
        PYBIND11_OVERRIDE_PURE(void, AbstractClient, OnNewObjectReceived, obj, id);
    }
};

// Forward declarations for submodules
void bind_utils(py::module &m);
void bind_projections(py::module &m);
void bind_clusterers(py::module &m);
void bind_ground_removal(py::module &m);
void bind_communication(py::module &m);
void bind_visualization(py::module &m);
void bind_image_labelers(py::module &m);


PYBIND11_MODULE(_depth_clustering, m) {
    m.doc() = "Python bindings for depth_clustering library";

    // Bind submodules
    bind_communication(m);
    bind_utils(m);
    bind_projections(m);
    bind_image_labelers(m);
    bind_clusterers(m);
    bind_ground_removal(m);
    bind_visualization(m);

}

std::shared_ptr<Cloud> numpy_to_cloud_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> points_nx3,
    py::array_t<double, py::array::c_style | py::array::forcecast> pitch_angles_rad,
    const std::tuple<double, double, double>& offset_xyz) {

    // Validation...
    if (points_nx3.ndim() != 2 || points_nx3.shape(1) != 3) {
        throw std::runtime_error("Input points must be an n×3 NumPy array.");
    }
    if (pitch_angles_rad.ndim() != 1) {
        throw std::runtime_error("Pitch angles must be a 1D NumPy array.");
    }

    const auto num_points = points_nx3.shape(0);
    const auto num_rings = pitch_angles_rad.shape(0);
    if (num_rings == 0) {
        throw std::runtime_error("Pitch angles array cannot be empty.");
    }

    const double* points_data = points_nx3.data();
    const double* angles_data = pitch_angles_rad.data();

    const double offset_x = std::get<0>(offset_xyz);
    const double offset_y = std::get<1>(offset_xyz);
    const double offset_z = std::get<2>(offset_xyz);

    // Create sorted index mapping
    std::vector<std::pair<double, int>> angle_index_pairs;
    angle_index_pairs.reserve(num_rings);
    for (ssize_t i = 0; i < num_rings; ++i) {
        angle_index_pairs.emplace_back(angles_data[i], i);
    }
    
    if(!std::is_sorted(angles_data, angles_data + num_rings)){
        // Sort by angle (keeps original indices)
        std::sort(angle_index_pairs.begin(), angle_index_pairs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    auto cloud = std::make_shared<Cloud>();
    cloud->reserve(num_points);

    for (ssize_t i = 0; i < num_points; ++i) {
        const size_t base_idx = i * 3;
        const double x = points_data[base_idx];
        const double y = points_data[base_idx + 1];
        const double z = points_data[base_idx + 2];
        
        const double rel_x = x - offset_x;
        const double rel_y = y - offset_y;
        const double rel_z = z - offset_z;

        const double xy_dist = std::hypot(rel_x, rel_y);
        const double point_pitch = std::atan2(rel_z, xy_dist);

        // Binary search on sorted angles
        auto it = std::lower_bound(angle_index_pairs.begin(), angle_index_pairs.end(), point_pitch,
                                   [](const auto& pair, double val) { return pair.first < val; });
        
        int best_ring_idx;
        if (it == angle_index_pairs.begin()) {
            best_ring_idx = it->second;  // Use original index
        } else if (it == angle_index_pairs.end()) {
            best_ring_idx = (it - 1)->second;
        } else {
            const double diff_curr = std::abs(it->first - point_pitch);
            const double diff_prev = std::abs((it - 1)->first - point_pitch);
            best_ring_idx = (diff_prev < diff_curr) ? (it - 1)->second : it->second;
        }

        cloud->push_back(RichPoint(
            static_cast<float>(x),
            static_cast<float>(y),
            static_cast<float>(z),
            static_cast<uint16_t>(best_ring_idx)
        ));
    }

    return cloud;
}

/**
 * @brief Binds various utility classes and functions to a Python module.
 *
 * This function creates a submodule named "utils" and populates it with bindings
 * for core data structures (like Radians, RichPoint, Cloud, Pose, Bbox),
 * file I/O helpers (FolderReader), and other miscellaneous utilities (Timer).
 *
 * @param m The main pybind11 module to which the submodule will be added.
 */
void bind_utils(py::module &m) {
    // Create the "utils" submodule with a descriptive docstring.
    py::module utils = m.def_submodule("utils", "Utility classes and functions");

    // Bind the C++ `Radians` class to a Python class of the same name.
    py::class_<Radians>(utils, "Radians")
        // Bind the default and parameterized constructors.
        .def(py::init<>())
        .def(py::init<Radians::IsRadians, float>())
        // Bind member functions for accessing values and properties.
        .def("val", &Radians::val)
        .def("valid", &Radians::valid)
        .def("ToDegrees", &Radians::ToDegrees)
        // Bind static member functions, which will be callable on the Python class itself.
        .def_static("FromRadians", &Radians::FromRadians)
        .def_static("FromDegrees", &Radians::FromDegrees)
        .def_static("Abs", &Radians::Abs)
        .def_static("Floor", &Radians::Floor)
        // Bind C++ operators to Python's special methods for rich comparison and arithmetic.
        // `py::self` is a placeholder for the class type.
        .def(py::self + py::self)      // Corresponds to __add__
        .def(py::self - py::self)      // Corresponds to __sub__
        .def(py::self += py::self)     // Corresponds to __iadd__ (in-place add)
        .def(py::self -= py::self)     // Corresponds to __isub__ (in-place sub)
        .def(py::self / float())       // Corresponds to __truediv__
        .def(py::self * float())       // Corresponds to __mul__
        .def(py::self < py::self)     // Corresponds to __lt__
        .def(py::self > py::self)      // Corresponds to __gt__
        // Bind overloaded C++ methods by explicitly casting the function pointer to the desired signature.
        .def("Normalize", (void (Radians::*)(const Radians&, const Radians&)) &Radians::Normalize)
        .def("Normalize", (Radians (Radians::*)(const Radians&, const Radians&) const) &Radians::Normalize);

    // Bind C++ user-defined literal operators as standalone functions in the Python module.
    utils.def("operator_rad", [](long double angle) { return Radians::FromRadians(angle); });
    utils.def("operator_deg", [](long double angle) { return Radians::FromDegrees(angle); });

    // Bind the `RichPoint` class, representing a single point with extra attributes.
    py::class_<RichPoint>(utils, "RichPoint")
        // Bind various constructors.
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def(py::init<float, float, float, uint16_t>())
        .def(py::init<Eigen::Vector3f&>())
        // Bind overloaded accessors. pybind11 automatically creates a read/write property in Python
        // when it sees a pair of `const` (getter) and non-`const` (setter) methods with the same name.
        .def("ring", (int (RichPoint::*)() const) &RichPoint::ring)
        .def("ring", (uint16_t& (RichPoint::*)()) &RichPoint::ring)
        .def("x", (float (RichPoint::*)() const) &RichPoint::x)
        .def("x", (float& (RichPoint::*)()) &RichPoint::x)
        .def("y", (float (RichPoint::*)() const) &RichPoint::y)
        .def("y", (float& (RichPoint::*)()) &RichPoint::y)
        .def("z", (float (RichPoint::*)() const) &RichPoint::z)
        .def("z", (float& (RichPoint::*)()) &RichPoint::z)
        .def("AsEigenVector", (const Eigen::Vector3f& (RichPoint::*)() const) &RichPoint::AsEigenVector)
        .def("AsEigenVector", (Eigen::Vector3f& (RichPoint::*)()) &RichPoint::AsEigenVector)
        // Bind other member functions.
        .def("DistToSensor2D", &RichPoint::DistToSensor2D)
        .def("DistToSensor3D", &RichPoint::DistToSensor3D)
        // Bind the equality operator.
        .def(py::self == py::self);

    // Use a pybind11 helper to bind `std::unordered_map<uint16_t, Cloud>` to a Python dict-like object.    
    py::bind_map<ClusterMap>(utils, "ClusterMap");

    // Bind Eigen::Affine3f. This is necessary because the Pose class inherits from it.
    py::class_<Eigen::Affine3f>(utils, "Affine3f")
        .def(py::init<>());

    // Bind the `Pose` class, specifying its base class `Eigen::Affine3f`.
    // This establishes the inheritance hierarchy in Python.
    py::class_<Pose, Eigen::Affine3f>(utils, "Pose")
        // Bind various constructors.
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def(py::init<const Eigen::Vector3f&>())
        .def(py::init<const Eigen::Affine3f&>())
        // Bind accessors and setters.
        .def("x", &Pose::x)
        .def("y", &Pose::y)
        .def("z", &Pose::z)
        .def("theta", &Pose::theta)
        .def("likelihood", &Pose::likelihood)
        .def("SetX", &Pose::SetX)
        .def("SetY", &Pose::SetY)
        .def("SetZ", &Pose::SetZ)
        .def("SetTheta", &Pose::SetTheta)
        .def("SetPitch", &Pose::SetPitch)
        .def("SetRoll", &Pose::SetRoll)
        .def("SetYaw", &Pose::SetYaw)
        .def("SetLikelihood", &Pose::SetLikelihood)
        // Bind transformation methods.
        .def("ToLocalFrameOf", &Pose::ToLocalFrameOf)
        // Bind overloaded methods by casting the function pointer.
        .def("InLocalFrameOf", (Pose (Pose::*)(const Pose&) const) &Pose::InLocalFrameOf)
        .def("InLocalFrameOf", (Pose::Ptr (Pose::*)(const Pose::Ptr&) const) &Pose::InLocalFrameOf)
        .def("Print2D", &Pose::Print2D)
        .def("Print3D", &Pose::Print3D)
        // Bind static factory and conversion methods.
        .def_static("FromVector6f", &Pose::FromVector6f)
        .def("ToVector6f", &Pose::ToVector6f);

    // Bind the `Cloud` class, specifying std::shared_ptr as the holder type for memory management.
    py::class_<Cloud, std::shared_ptr<Cloud>>(utils, "Cloud")
        // Bind various constructors.
        .def(py::init<>())
        .def(py::init<const Cloud&>())
        .def(py::init<const Pose&>())
        // Bind properties and member functions.
        .def("points", &Cloud::points)
        .def("pose", (Pose& (Cloud::*)()) &Cloud::pose)
        .def("pose", (const Pose& (Cloud::*)() const) &Cloud::pose)
        .def("sensor_pose", (Pose& (Cloud::*)()) &Cloud::sensor_pose)
        .def("sensor_pose", (const Pose& (Cloud::*)() const) &Cloud::sensor_pose)
        // Bind methods that mirror Python list-like behavior.
        .def("push_back", &Cloud::push_back)
        .def("size", &Cloud::size)
        .def("empty", &Cloud::empty)
        .def("reserve", &Cloud::reserve)
        // Bind operator[] to Python's __getitem__ to allow bracket access (e.g., `cloud[i]`).
        .def("__getitem__", (RichPoint& (Cloud::*)(int)) &Cloud::operator[])
        .def("__getitem__", (const RichPoint& (Cloud::*)(int) const) &Cloud::operator[])
        // Bind `at` method for bounds-checked access.
        .def("at", (RichPoint& (Cloud::*)(int)) &Cloud::at)
        .def("at", (const RichPoint& (Cloud::*)(int) const) &Cloud::at)
        .def("Resize", &Cloud::Resize)
        .def("SetPose", &Cloud::SetPose)
        // This lambda function is a custom wrapper to bridge the gap between C++ boost::shared_ptr
        // and Python's std::shared_ptr (which pybind11 uses).
        .def("projection_ptr", [](const Cloud& self) -> std::shared_ptr<const CloudProjection> {
            auto boost_ptr = self.projection_ptr();  // C++ method returns a boost::shared_ptr
            if (!boost_ptr) {
                return nullptr;
            }
            // Create a std::shared_ptr that takes ownership. The custom deleter ensures that when
            // the std::shared_ptr is destroyed, the original boost_ptr is also reset correctly.
            return std::shared_ptr<const CloudProjection>(
                boost_ptr.get(),
                [boost_ptr](const CloudProjection*) mutable {
                    boost_ptr.reset();
                }
            );
        })
        // A similar wrapper for the non-const (mutable) version of the projection pointer.
        .def("projection_ptr_mut", [](Cloud& self) -> std::shared_ptr<CloudProjection> {
            auto boost_ptr = self.projection_ptr(); 
            if (!boost_ptr) {
                return nullptr;
            }
            return std::shared_ptr<CloudProjection>(
                boost_ptr.get(),
                [boost_ptr](CloudProjection*) mutable {
                    boost_ptr.reset();
                }
            );
        })
        .def("PointsProjectedToPixel", &Cloud::PointsProjectedToPixel)
        .def("TransformInPlace", &Cloud::TransformInPlace)
        .def("Transform", &Cloud::Transform)
        // A wrapper for the setter, performing the inverse conversion: from std::shared_ptr (Python)
        // to boost::shared_ptr (for the C++ method).
        .def("SetProjectionPtr", [](Cloud& self, std::shared_ptr<CloudProjection> proj) {
            // Create a boost::shared_ptr that shares ownership with the incoming std::shared_ptr.
            boost::shared_ptr<CloudProjection> boost_proj(
                proj.get(),
                [proj](CloudProjection*) mutable { proj.reset(); }
            );
            // Call the C++ method with the correctly typed smart pointer.
            self.SetProjectionPtr(boost_proj);
        }, py::arg("projection"))
        .def("InitProjection", &Cloud::InitProjection)
        .def_static("FromImage", &Cloud::FromImage)
    // Use conditional compilation to only bind PCL-related functions if PCL was found during the build.
    #ifdef PCL_FOUND
        .def("ToPcl", &Cloud::ToPcl)
        .def_static("FromPcl", &Cloud::FromPcl<pcl::PointXYZ>)
    #endif
        ;

    // Bind the `Bbox` (bounding box) class.
    py::class_<Bbox>(utils, "Bbox")
        .def(py::init<>())
        .def(py::init<const Cloud&>())
        .def(py::init<const Eigen::Vector3f&, const Eigen::Vector3f&>())
        .def("Intersects", &Bbox::Intersects)
        .def("Intersect", &Bbox::Intersect)
        .def("volume", &Bbox::volume)
        .def("center", &Bbox::center)
        .def("scale", &Bbox::scale)
        .def("min_point", &Bbox::min_point)
        .def("max_point", &Bbox::max_point)
        .def("GetScaleX", &Bbox::GetScaleX)
        .def("GetScaleY", &Bbox::GetScaleY)
        .def("GetScaleZ", &Bbox::GetScaleZ)
        .def("MoveBy", &Bbox::MoveBy)
        .def("UpdateScaleAndCenter", &Bbox::UpdateScaleAndCenter);

    // Bind the `FolderReader::Order` C++ enum to a Python enum.
    py::enum_<FolderReader::Order>(utils, "Order")
        .value("SORTED", FolderReader::Order::SORTED)
        .value("UNDEFINED", FolderReader::Order::UNDEFINED);


    // Bind the `FolderReader` class for file system interaction.
    py::class_<FolderReader>(utils, "FolderReader")
        // Bind constructors, using py::arg to name arguments in Python and provide default values.
        .def(py::init<const std::string&, const std::string&, FolderReader::Order>(),
             py::arg("folder_path"), py::arg("ending_with"), py::arg("order") = FolderReader::Order::UNDEFINED)
        .def(py::init<const std::string&, const std::string&, const std::string&, FolderReader::Order>(),
             py::arg("folder_path"), py::arg("starting_with"), py::arg("ending_with"), py::arg("order") = FolderReader::Order::UNDEFINED)
        .def("GetNextFilePath", &FolderReader::GetNextFilePath)
        .def("GetAllFilePaths", &FolderReader::GetAllFilePaths);

    // VelodyneUtils functions
    utils.def("CloudFromMat", &CloudFromMat);
    utils.def("ReadKittiCloud", &ReadKittiCloud, "Reads a point cloud from a KITTI .bin file.", py::arg("path"));
    utils.def("ReadKittiCloudTxt", &ReadKittiCloudTxt);
    utils.def("MatFromDepthPng", &MatFromDepthPng);

    // Bind the `Timer::Units` enum.
    py::enum_<time_utils::Timer::Units>(utils, "TimerUnits")
        .value("Micro", time_utils::Timer::Units::Micro)
        .value("Milli", time_utils::Timer::Units::Milli);

    // Bind the `Timer` class for performance measurement.
    py::class_<time_utils::Timer>(utils, "Timer")
        .def(py::init<>())
        .def("start", &time_utils::Timer::start)
        .def("measure", &time_utils::Timer::measure, py::arg("units") = time_utils::Timer::Units::Micro);

    utils.def("numpy_to_cloud", &numpy_to_cloud_cpp,
    "A fast, C++ implementation to convert a NumPy point cloud to a dc::utils::Cloud object.",
    py::arg("points_nx3"),
    py::arg("pitch_angles_rad"),
    py::arg("offset_xyz")
    );

    // Create a nested submodule for memory-related utilities.
    py::module mem_utils = utils.def_submodule("mem_utils", "Memory utilities");
    // Conditionally bind a function that is only available if QT is found.
    #ifdef QT_FOUND
    // This binds a specific instantiation of the `make_unique` template function.
    mem_utils.def("make_unique", &mem_utils::make_unique<DrawableCube, const Eigen::Vector3f&, const Eigen::Vector3f&, const Eigen::Vector3f&>);
    #endif
}

/**
 * @brief Binds C++ projection classes and functions to a Python module.
 *
 * This function creates a submodule named "projections" and populates it with
 * Python bindings for C++ classes related to point cloud projections, such as
 * SpanParams, ProjectionParams, CloudProjection, and its derivatives.
 *
 * @param m The main pybind11 module to which the submodule will be added.
 */
void bind_projections(py::module &m) {
    // Create a submodule named "projections" with a docstring.
    // All subsequent bindings will be within this submodule.
    py::module projections = m.def_submodule("projections", "Projection classes and functions");

    // Bind the SpanParams C++ class to a Python class named "SpanParams".
    // This class defines the parameters for a 1D scan, like a single laser ring or a row/column in an image.
    py::class_<SpanParams>(projections, "SpanParams")
        // Bind the default constructor.
        .def(py::init<>())
        // Bind a constructor that initializes the span using start/end angles and a step angle.
        .def(py::init<const Radians&, const Radians&, const Radians&>(), 
             py::arg("start_angle"), py::arg("end_angle"), py::arg("step"))
        // Bind an alternative constructor that uses the number of beams instead of a step angle.
        .def(py::init<const Radians&, const Radians&, int>(), 
             py::arg("start_angle"), py::arg("end_angle"), py::arg("num_beams"))
        // Bind accessor methods for the span parameters.
        // The return value policy 'copy' ensures Python gets a new copy of the value.
        .def("start_angle", &SpanParams::start_angle, py::return_value_policy::copy)
        .def("end_angle", &SpanParams::end_angle, py::return_value_policy::copy)
        .def("step", &SpanParams::step, py::return_value_policy::copy)
        .def("span", &SpanParams::span, py::return_value_policy::copy)
        .def("num_beams", &SpanParams::num_beams)
        // Bind a method to check if the span parameters are valid.
        .def("valid", &SpanParams::valid);

    // Bind the C++ enum SpanParams::Direction to a Python enum named "SpanDirection".
    // This enum specifies whether a span is horizontal or vertical.
    py::enum_<SpanParams::Direction>(projections, "SpanDirection")
        .value("HORIZONTAL", SpanParams::Direction::HORIZONTAL)
        .value("VERTICAL", SpanParams::Direction::VERTICAL)
        // .export_values() makes the enum members accessible from the module's scope
        // (e.g., projections.HORIZONTAL) as well as the enum's scope (projections.SpanDirection.HORIZONTAL).
        .export_values();

    // Bind the ProjectionParams C++ class. This class holds the full configuration for a 2D projection.
    // It is managed by a std::shared_ptr, which is a common pattern for classes that might be part of a polymorphic hierarchy.
    py::class_<ProjectionParams, std::shared_ptr<ProjectionParams>>(projections, "ProjectionParams")
        // Bind the default constructor.
        .def(py::init<>())
        // Bind the SetSpan method. The C++ function signature is explicitly cast to resolve potential overloads.
        // This version sets the span parameters for a given direction using a single SpanParams object.
        .def("SetSpan", (void (ProjectionParams::*)(const SpanParams&, const SpanParams::Direction&)) &ProjectionParams::SetSpan,
             py::arg("span_params"), py::arg("direction"))
        // Bind the overloaded SetSpan method that takes a vector of SpanParams.
        // This is useful for sensors with non-uniform beam spacing (e.g., most Velodyne LiDARs).
        .def("SetSpan", (void (ProjectionParams::*)(const std::vector<SpanParams>&, const SpanParams::Direction&)) &ProjectionParams::SetSpan,
             py::arg("span_params_list"), py::arg("direction"))
        // Bind accessor methods for projection properties.
        .def("rows", &ProjectionParams::rows)
        .def("cols", &ProjectionParams::cols)
        .def("size", &ProjectionParams::size)
        .def("valid", &ProjectionParams::valid)
        // Bind static factory methods to easily create pre-configured parameters for common LiDAR models.
        .def_static("VLP_16", &ProjectionParams::VLP_16)
        .def_static("HDL_32", &ProjectionParams::HDL_32)
        .def_static("HDL_64", &ProjectionParams::HDL_64)
        .def_static("HDL_64_EQUAL", &ProjectionParams::HDL_64_EQUAL)
        // Bind a static factory method to load parameters from a configuration file.
        .def_static("FromConfigFile", &ProjectionParams::FromConfigFile, py::arg("path"))
        // Bind a static factory method for a full spherical projection with a given discretization.
        // The argument has a default value specified in C++.
        .def_static("FullSphere", &ProjectionParams::FullSphere, py::arg("discretization") = 5.0_deg);

    // Bind the CloudProjection C++ class. This is the base class for all projection types.
    // `PyCloudProjection` is a trampoline class that allows Python classes to inherit from CloudProjection
    // and override its virtual methods. `std::shared_ptr` is used for memory management.
    py::class_<CloudProjection, PyCloudProjection, std::shared_ptr<CloudProjection>>(projections, "CloudProjection")
        // Bind the constructor, which takes projection parameters.
        .def(py::init<const ProjectionParams&>(), py::arg("params"))
        // Bind the method to project a vector of points into the projection's internal representation (e.g., a depth image).
        .def("InitFromPoints", (void (CloudProjection::*)(const RichPoint::AlignedVector&)) &CloudProjection::InitFromPoints, py::arg("points"))
        // Bind the accessor for the projection parameters. `return_value_policy::reference` avoids a copy.
        .def("params", &CloudProjection::params, py::return_value_policy::reference)
        // Bind the accessor for the resulting depth image (as a cv::Mat).
        // `return_value_policy::reference_internal` is crucial: it tells pybind11 that the lifetime
        // of the returned cv::Mat is tied to the parent CloudProjection object, preventing dangling references.
        .def("depth_image", (const cv::Mat& (CloudProjection::*)() const) &CloudProjection::depth_image, py::return_value_policy::reference_internal)
        // Bind the Clone() method. The C++ version likely returns a custom smart pointer (e.g., boost::shared_ptr).
        // This lambda function wraps the custom smart pointer in a std::shared_ptr that pybind11 understands,
        // ensuring correct memory management across the C++/Python boundary.
        .def("Clone", [](const CloudProjection& self) {
            auto boost_ptr = self.Clone();
            // Create a std::shared_ptr that shares ownership with the original boost_ptr. 
            return std::shared_ptr<CloudProjection>(
                boost_ptr.get(), 
                // The custom deleter ensures that when the std::shared_ptr is destroyed,
                // the original boost_ptr is also reset, releasing the object.
                [boost_ptr](CloudProjection*) mutable {
                    boost_ptr.reset();
                }
            );
        });

    // Bind the RingProjection class, which inherits from CloudProjection.
    // This establishes the inheritance relationship in Python, so isinstance() works as expected.
    py::class_<RingProjection, CloudProjection, std::shared_ptr<RingProjection>>(projections, "RingProjection")
        // Bind the Clone method for this derived class, using the same smart pointer conversion pattern.
        .def(py::init<const ProjectionParams&>(), py::arg("params"))
        .def("Clone", [](const RingProjection& self) {
            auto boost_ptr = self.Clone();
            return std::shared_ptr<CloudProjection>(
                boost_ptr.get(), 
                [boost_ptr](CloudProjection*) mutable {
                    boost_ptr.reset();
                }
            );
        });

    // Bind the SphericalProjection class, another derivative of CloudProjection.
    py::class_<SphericalProjection, CloudProjection, std::shared_ptr<SphericalProjection>>(projections, "SphericalProjection")
        .def(py::init<const ProjectionParams&>(), py::arg("params"))
        // Bind the Clone method for this derived class, again using the same pattern.
        .def("Clone", [](const SphericalProjection& self) {
            auto boost_ptr = self.Clone();
            return std::shared_ptr<CloudProjection>(
                boost_ptr.get(), 
                [boost_ptr](CloudProjection*) mutable {
                    boost_ptr.reset();
                }
            );
        });
}

/**
 * @brief Binds C++ clustering algorithm classes to a Python module.
 *
 * This function creates a submodule named "clusterers" and exposes various C++
 * clustering classes, including the abstract base class `AbstractClusterer` and
 * concrete implementations like `ImageBasedClusterer` and `EuclideanClusterer`.
 *
 * @param m The main pybind11 module to which the submodule will be added.
 */
void bind_clusterers(py::module &m) {
    // Create a submodule named "clusterers" with a descriptive docstring.
    py::module clusterers = m.def_submodule("clusterers", "Clustering algorithms");

    // Define type aliases for clarity and brevity in the bindings.
    // This makes the py::class_ declarations easier to read and maintain.
    using PyImageBasedClusterer = ImageBasedClusterer<LinearImageLabeler<1, 1>>;
    using Clusters = std::unordered_map<uint16_t, Cloud>;
    using DepthMap = cv::Mat;

    // Bind the abstract base class `AbstractClusterer`.
    // It's crucial to bind base classes so that pybind11 understands the inheritance hierarchy.
    // This allows Python code to correctly handle polymorphism (e.g., passing a concrete
    // clusterer where an abstract one is expected).
    // Template arguments specify:
    // 1. `AbstractClusterer`: The C++ class to bind.
    // 2. `AbstractClient<Cloud>`: The first base class.
    // 3. `AbstractSender<ClusterMap>`: The second base class.
    // 4. `std::shared_ptr<AbstractClusterer>`: The holder type, indicating memory is managed by std::shared_ptr.
    py::class_<AbstractClusterer,
               AbstractClient<Cloud>,          
               AbstractSender<ClusterMap>,   
               std::shared_ptr<AbstractClusterer>>(clusterers, "AbstractClusterer");

    // Bind the concrete `ImageBasedClusterer` class (using the `PyImageBasedClusterer` alias).
    // Template arguments specify:
    // 1. `PyImageBasedClusterer`: The C++ class to bind.
    // 2. `std::shared_ptr<PyImageBasedClusterer>`: Its specific shared_ptr holder type.
    // 3. `AbstractClusterer`: Its direct parent class in the C++ hierarchy.
    // This declaration tells pybind11 that `ImageBasedClusterer` in Python inherits from `AbstractClusterer`.
    py::class_<PyImageBasedClusterer,
               std::shared_ptr<PyImageBasedClusterer>, 
               AbstractClusterer>                     
        (clusterers, "ImageBasedClusterer")


        // Bind the constructor that takes only the angle tolerance.
        // This maps to the Python `__init__` method.
        .def(py::init<const Radians&>(),
             py::arg("angle_tollerance")) // `py::arg` provides a named argument in Python.
        // Bind the full constructor with all parameters for more control.
        .def(py::init<Radians, uint16_t, uint16_t, uint16_t>(),
             py::arg("angle_tollerance"),
             py::arg("min_cluster_size"),
             py::arg("max_cluster_size"),
             py::arg("skip"))
        // Bind the `OnNewObjectReceived` method. Because the base classes were declared,
        // pybind11 can automatically resolve inherited methods. Explicitly binding them
        // allows us to name arguments and add documentation.
        .def("OnNewObjectReceived", &PyImageBasedClusterer::OnNewObjectReceived,
             py::arg("cloud"), py::arg("sender_id") = 0)
        // Bind the `AddClient` method, which is inherited from `AbstractSender`.
        // The `py::keep_alive<1, 2>()` call policy is critical for memory management.
        // It creates a reference that keeps the second argument (the `client`, index 2)
        // alive as long as the first argument (the `self` object, index 1) is alive.
        // This prevents the Python garbage collector from destroying the client object
        // while the C++ clusterer still holds a pointer to it.
        .def("AddClient", &PyImageBasedClusterer::AddClient, py::arg("client"),
             "Adds a client to the list of receivers. Use py::keep_alive to manage memory.",
             py::keep_alive<1, 2>())
        // Bind the method to set the difference type for clustering.
        .def("SetDiffType", &PyImageBasedClusterer::SetDiffType, py::arg("diff_type"))     
         // Bind the method to set a client for receiving labeled images.
        // The `py::keep_alive<1, 2>()` policy is used here for the same reason as in `AddClient`,
        // ensuring the image client's lifetime is tied to the clusterer.       
        .def("SetLabelImageClient", &PyImageBasedClusterer::SetLabelImageClient, py::arg("client"),
             "Sets the client to receive color images with labels. Use keep_alive to ensure the client is not garbage-collected.",
             py::keep_alive<1, 2>());


    // Bind the `EuclideanClusterer` class, another concrete implementation inheriting from `AbstractClusterer`.
    py::class_<EuclideanClusterer,
               std::shared_ptr<EuclideanClusterer>,
               AbstractClusterer>
        (clusterers, "EuclideanClusterer")
        // Bind its constructor, providing default values for all arguments.
        // This makes the clusterer easy to instantiate from Python with minimal configuration.
        .def(py::init<double, uint16_t, uint16_t, uint16_t>(),
             py::arg("cluster_tollerance") = 0.2,
             py::arg("min_cluster_size") = 100,
             py::arg("max_cluster_size") = 25000,
             py::arg("skip") = 10)
        // Bind its specific implementation of the `OnNewObjectReceived` method.
        .def("OnNewObjectReceived", &EuclideanClusterer::OnNewObjectReceived,
             py::arg("cloud"), py::arg("sender_id") = 0);
}


/**
 * @brief Binds C++ ground removal classes and functions to a Python module.
 *
 * This function creates a submodule named "ground_removal" and exposes the
 * C++ `DepthGroundRemover` class to Python, allowing it to be instantiated
 * and used within Python scripts.
 *
 * @param m The main pybind11 module to which the submodule will be added.
 */
void bind_ground_removal(py::module &m) {
    // Create a submodule named "ground_removal" with a descriptive docstring.
    // All subsequent bindings within this function will belong to this submodule.
    py::module ground_removal = m.def_submodule("ground_removal", "Ground removal algorithms");

    // Bind the C++ class `DepthGroundRemover` to a Python class of the same name.
    // The template arguments specify:
    // 1. `DepthGroundRemover`: The C++ class being bound.
    // 2. `std::shared_ptr<DepthGroundRemover>`: The holder type. This tells pybind11
    //    that instances of this class are managed by std::shared_ptr on the C++ side,
    //    ensuring proper memory management across the language boundary.
    // 3. `AbstractSender<Cloud>`: A public base class of DepthGroundRemover.
    // 4. `AbstractClient<Cloud>`: Another public base class.
    // Declaring base classes is crucial for enabling polymorphism and ensuring that
    // pybind11 can correctly cast pointers between base and derived types.
    py::class_<DepthGroundRemover,
               std::shared_ptr<DepthGroundRemover>,
               AbstractSender<Cloud>,
               AbstractClient<Cloud> >(ground_removal, "DepthGroundRemover")
        // Bind the C++ constructor to the Python `__init__` method.
        // It takes projection parameters, a ground removal angle, and an optional window size.
        .def(py::init<const ProjectionParams&, const Radians&, int>(),
             // Use py::arg to name the arguments in Python for clarity and keyword argument support.
             py::arg("params"),
             py::arg("ground_remove_angle"),
             py::arg("window_size") = 5) // The C++ default value is exposed to Python.
        // Bind the `OnNewObjectReceived` method.
        // A lambda function is used here to adapt the C++ interface for Python.
        .def("OnNewObjectReceived", [](DepthGroundRemover& self, const Cloud& cloud, int sender_id) {
            // Call the actual C++ member function on the instance (`self`).
            self.OnNewObjectReceived(cloud, sender_id);
            // The lambda has a `void` return type, so the corresponding Python method will return `None`.
            // This is useful if the C++ method returns a value that is not needed or not easily convertible in Python.
        }, py::arg("cloud"), py::arg("sender_id") = 0); // Name the arguments and provide a default value for sender_id.
}


// Wrapper for AbstractSender<Cloud>::AddClient
void AddClient_Cloud_Wrapper(AbstractSender<Cloud>& sender, std::shared_ptr<AbstractClient<Cloud>> client) {
    sender.AddClient(client.get());
}

// Wrapper for AbstractSender<ClusterMap>::AddClient
void AddClient_ClusterMap_Wrapper(AbstractSender<ClusterMap>& sender, std::shared_ptr<AbstractClient<ClusterMap>> client) {
    sender.AddClient(client.get());
}

// Wrapper for AbstractSender<DepthMap>::AddClient
void AddClient_DepthMap_Wrapper(AbstractSender<DepthMap>& sender, std::shared_ptr<AbstractClient<DepthMap>> client) {
    sender.AddClient(client.get());
}

/**
 * @brief Binds the C++ communication interfaces (Sender/Client pattern) to a Python module.
 *
 * This function creates a "communication" submodule and exposes the generic
 * `AbstractClient` and `AbstractSender` template classes for various data types
 * like `Cloud`, `ClusterMap`, etc. This allows creating and connecting C++ and
 * Python components in a flexible pipeline.
 *
 * @param m The main pybind11 module to which the submodule will be added.
 */
void bind_communication(py::module &m) {
    // Create a submodule for all communication-related bindings.
    py::module communication = m.def_submodule("communication", "Communication interfaces (sender/client)");

    // --- Client Bindings ---
    // These bindings expose the `AbstractClient<T>` base classes to Python.
    // They are intended to be inherited from in Python to create custom processing nodes.

    // Bind `AbstractClient<Cloud>`.
    // The template arguments are:
    // 1. `AbstractClient<Cloud>`: The C++ class being bound.
    // 2. `PyClient<Cloud>`: A "trampoline" class. This is essential. It allows Python classes
    //    that inherit from this binding to override C++ virtual functions (like `OnNewObjectReceived`).
    //    When a C++ function calls the virtual method, the trampoline redirects the call to the Python implementation.
    // 3. `std::shared_ptr<AbstractClient<Cloud>>`: The holder type for memory management.
    py::class_<AbstractClient<Cloud>, PyClient<Cloud>, std::shared_ptr<AbstractClient<Cloud>>>
        (communication, "PyAbstractClient_Cloud").def(py::init<>());

    // Bind `AbstractClient<ClusterMap>` using the same trampoline pattern.
    py::class_<AbstractClient<ClusterMap>, PyClient<ClusterMap>, std::shared_ptr<AbstractClient<ClusterMap>>>
        (communication, "PyAbstractClient_ClusterMap").def(py::init<>());

    // Create a Python alias. `AbstractClient_Clusters` will be another name for `PyAbstractClient_ClusterMap`.
    // This can be used for convenience or backward compatibility.    
    communication.attr("AbstractClient_Clusters") = communication.attr("PyAbstractClient_ClusterMap");


    // Bind `AbstractClient<DepthMap>` (cv::Mat) using the trampoline pattern.
    py::class_<AbstractClient<DepthMap>, PyClient<DepthMap>, std::shared_ptr<AbstractClient<DepthMap>>>
        (communication, "PyAbstractClient_DepthMap").def(py::init<>());

    // Bind `AbstractClient<cv::Mat>` for generic OpenCV matrices.
    py::class_<AbstractClient<cv::Mat>, PyClient<cv::Mat>, std::shared_ptr<AbstractClient<cv::Mat>>>
        (communication, "PyAbstractClient_Mat").def(py::init<>());
    
    // --- Sender Bindings ---
    // These bindings expose the `AbstractSender<T>` base classes. C++ objects that inherit from
    // this (like clusterers) can then have Python clients attached to them.

    // Bind `AbstractSender<Cloud>`. No trampoline class is needed here because we don't expect
    // to inherit from `AbstractSender` in Python, but rather to call its methods.
    py::class_<AbstractSender<Cloud>, std::shared_ptr<AbstractSender<Cloud>>>
        (communication, "AbstractSender_Cloud")
        // Bind the `AddClient` method. Instead of binding the member function directly, a wrapper
        // function `AddClient_Cloud_Wrapper` is used. This wrapper likely handles the necessary
        // type casting from the Python client object to the `AbstractClient<Cloud>*` that C++ expects.
        .def("AddClient", &AddClient_Cloud_Wrapper, 
             "Adds a client to the sender.", 
             py::arg("client"),
             // The `keep_alive` policy is CRITICAL. It tells pybind11 to tie the lifetime of
             // argument #2 (the client) to the lifetime of argument #1 (the 'self' object, i.e., the sender).
             // This prevents the Python garbage collector from destroying the client object while the
             // C++ sender still holds a raw pointer to it, avoiding a dangling pointer and crash.
             py::keep_alive<1, 2>()); 

    // Bind `AbstractSender<ClusterMap>` with its corresponding wrapper and keep_alive policy.
    py::class_<AbstractSender<ClusterMap>, std::shared_ptr<AbstractSender<ClusterMap>>>
        (communication, "AbstractSender_ClusterMap")
        .def("AddClient", &AddClient_ClusterMap_Wrapper, 
             "Adds a client to the sender.", 
             py::arg("client"),
             py::keep_alive<1, 2>());

    // Bind `AbstractSender<DepthMap>` with its wrapper and keep_alive policy.             
    py::class_<AbstractSender<DepthMap>, std::shared_ptr<AbstractSender<DepthMap>>>
        (communication, "AbstractSender_DepthMap")
        .def("AddClient", &AddClient_DepthMap_Wrapper,
             "Adds a client to the sender.",
             py::arg("client"),
             py::keep_alive<1, 2>());

    // Bind `AbstractSender<cv::Mat>`.
    py::class_<AbstractSender<cv::Mat>, std::shared_ptr<AbstractSender<cv::Mat>>>
        (communication, "AbstractSender_Mat")
        // Here, a lambda is used instead of a named wrapper function. The effect is the same:
        // it takes the sender (`self`) and the client as arguments and calls the C++ `AddClient` method.
        .def("AddClient", [](AbstractSender<cv::Mat>* self, AbstractClient<cv::Mat>* client) {
            self->AddClient(client);
        }, "Adds a client to the sender.", 
           py::arg("client"),
           py::keep_alive<1, 2>());// The essential keep_alive policy is applied here as well.
}



/**
 * @brief Binds visualization and data-saving utility classes to a Python module.
 *
 * This function creates a "visualization" submodule and exposes several "saver" classes.
 * These classes act as sinks in a data processing pipeline, designed to receive data
 * (like point clouds or depth maps) and save them to the disk. They inherit from
 * the appropriate `AbstractClient` types, allowing them to be connected to `AbstractSender`
 * nodes in a pipeline constructed in Python.
 *
 * @param m The main pybind11 module to which the submodule will be added.
 */
void bind_visualization(py::module &m) {
    // Create a submodule named "visualization" with a descriptive docstring.
    py::module visualization = m.def_submodule("visualization", "Visualization classes");

    // Define type aliases for the client base classes. This improves readability and ensures
    // consistency with the aliases used in other binding functions (like bind_communication).
    using CloudClient = AbstractClient<Cloud>;
    using ClustersClient = AbstractClient<ClusterMap>; 
    using DepthMap = std::unordered_map<uint16_t, cv::Mat>;// Alias for the depth map data type.
    using DepthMapClient = AbstractClient<DepthMap>;

    // Bind the CloudSaver class, which saves single point clouds.
    // It is managed by a std::shared_ptr and inherits from CloudClient.
    py::class_<CloudSaver, std::shared_ptr<CloudSaver>, CloudClient>(visualization, "CloudSaver")
        // Bind the constructor, which takes a string prefix for the output filenames.
        .def(py::init<const std::string&>(), py::arg("prefix"))
        // Bind the core pipeline method. When a Cloud is sent to this object,
        // this method is invoked to handle saving it.
        .def("OnNewObjectReceived", &CloudSaver::OnNewObjectReceived);

    // Bind the VectorCloudSaver class, which saves clusters (maps of clouds).
    // It inherits from ClustersClient to receive ClusterMap objects.
    py::class_<VectorCloudSaver, std::shared_ptr<VectorCloudSaver>, ClustersClient>(visualization, "VectorCloudSaver")
        // Bind the constructor, taking a filename prefix and a sampling rate (`save_every`).
        .def(py::init<const std::string&, const size_t>(), py::arg("prefix"), py::arg("save_every") = 1)
        // Bind the pipeline method for receiving and saving cluster maps.
        .def("OnNewObjectReceived", &VectorCloudSaver::OnNewObjectReceived);

    // Bind the DepthMapSaver class, which saves depth map images.
    // It inherits from DepthMapClient to receive DepthMap objects.
    py::class_<DepthMapSaver, std::shared_ptr<DepthMapSaver>, DepthMapClient>(visualization, "DepthMapSaver")
        // Bind the constructor, taking a filename prefix and a sampling rate.
        .def(py::init<const std::string&, size_t>(), py::arg("prefix"), py::arg("save_every") = 1)
        // Bind the pipeline method for receiving and saving depth maps.
        .def("OnNewObjectReceived", &DepthMapSaver::OnNewObjectReceived);
}



/**
 * @brief Binds C++ image labeling algorithms and related data structures to Python.
 *
 * This function creates an "image_labelers" submodule. It exposes classes for
 * connected-component labeling on depth images, including the core algorithm
 * (`LinearImageLabeler`), helper data structures (`PixelCoord`, `HashQueue`),
 * and a factory system for different pixel comparison strategies (`DiffFactory`
 * and various `AbstractDiff` implementations).
 *
 * @param m The main pybind11 module to which the submodule will be added.
 */
void bind_image_labelers(py::module &m) {
    // Create the "image_labelers" submodule with a descriptive docstring.
    py::module image_labelers = m.def_submodule("image_labelers", "Image labeling algorithms");

    // Bind the PixelCoord class, a simple struct for holding (row, col) coordinates.
    py::class_<PixelCoord>(image_labelers, "PixelCoord")
        .def(py::init<>())// Default constructor
        .def(py::init<int16_t, int16_t>())// Constructor from row and col
        // Expose public members as read-only properties using lambda functions.
        .def("row", [](PixelCoord& self) { return self.row; })
        .def("col", [](PixelCoord& self) { return self.col; })
        // Bind the addition operator to allow `coord1 + coord2` in Python.
        .def(py::self + py::self);

    // Bind the HashQueue class, a specialized queue with fast "contains" checks.
    py::class_<HashQueue>(image_labelers, "HashQueue")
        // Bind the constructor, which can take an estimated size for pre-allocation.
        .def(py::init<int>(), py::arg("estimated_size") = 0)
        // Bind the standard queue operations.
        .def("push", &HashQueue::push)
        .def("pop", &HashQueue::pop)
        .def("empty", &HashQueue::empty)
        .def("front", &HashQueue::front)
        // Bind the special method for checking if an element is in the queue.
        .def("contains", &HashQueue::contains);

    // Bind the AbstractImageLabeler class, which defines the interface for labelers.
    py::class_<AbstractImageLabeler>(image_labelers, "AbstractImageLabeler")
        .def("SetDepthImage", &AbstractImageLabeler::SetDepthImage)
        .def("ComputeLabels", &AbstractImageLabeler::ComputeLabels)
        .def("GetLabelImage", &AbstractImageLabeler::GetLabelImage)
        // Bind the static method for converting a label image to a color visualization.
        .def_static("LabelsToColor", &AbstractImageLabeler::LabelsToColor);

    // Bind the LinearImageLabeler class, a concrete implementation of the labeler.
    // This is a template specialization for a step size of 1 in both row and column.
    // The second template argument specifies its base class for Python's inheritance system.
    py::class_<LinearImageLabeler<1, 1>, AbstractImageLabeler>(image_labelers, "LinearImageLabeler")
        // Bind the constructor using a lambda. This is a common pattern for creating
        // instances of templated classes or when custom logic is needed.
        .def(py::init([](const cv::Mat& mat, const ProjectionParams& params, const Radians& radians) {
            return new LinearImageLabeler<1, 1>(mat, params, radians);
        }))
        // Bind the public member functions of the class.
        .def("ComputeLabels", &LinearImageLabeler<1, 1>::ComputeLabels)
        .def("LabelOneComponent", &LinearImageLabeler<1, 1>::LabelOneComponent)
        .def("DepthAt", &LinearImageLabeler<1, 1>::DepthAt)
        .def("LabelAt", &LinearImageLabeler<1, 1>::LabelAt)
        .def("SetLabel", &LinearImageLabeler<1, 1>::SetLabel)
        .def("WrapCols", &LinearImageLabeler<1, 1>::WrapCols);

    // Bind the DiffFactory::DiffType C++ enum to a Python enum object.
    py::enum_<DiffFactory::DiffType>(image_labelers, "DiffType")
        .value("NONE", DiffFactory::DiffType::NONE)
        .value("SIMPLE", DiffFactory::DiffType::SIMPLE)
        .value("ANGLES", DiffFactory::DiffType::ANGLES)
        .value("ANGLES_PRECOMPUTED", DiffFactory::DiffType::ANGLES_PRECOMPUTED)
        .value("LINE_DIST", DiffFactory::DiffType::LINE_DIST)
        .value("LINE_DIST_PRECOMPUTED", DiffFactory::DiffType::LINE_DIST_PRECOMPUTED);

    // Bind the AbstractDiff class, the interface for pixel difference calculators.
    py::class_<AbstractDiff>(image_labelers, "AbstractDiff")
        .def("DiffAt", &AbstractDiff::DiffAt)
        .def("SatisfiesThreshold", &AbstractDiff::SatisfiesThreshold)
        .def("Visualize", &AbstractDiff::Visualize);

    // Bind the SimpleDiff class, inheriting from AbstractDiff
    py::class_<SimpleDiff, AbstractDiff>(image_labelers, "SimpleDiff")
        .def(py::init<const cv::Mat*>())
        .def("DiffAt", &SimpleDiff::DiffAt)
        .def("SatisfiesThreshold", &SimpleDiff::SatisfiesThreshold);

    // Bind the AngleDiff class, inheriting from AbstractDiff.
    py::class_<AngleDiff, AbstractDiff>(image_labelers, "AngleDiff")
        .def(py::init<const cv::Mat*, const ProjectionParams*>())
        .def("DiffAt", &AngleDiff::DiffAt)
        .def("SatisfiesThreshold", &AngleDiff::SatisfiesThreshold)
        .def("Visualize", &AngleDiff::Visualize);

    // Bind the AngleDiffPrecomputed class, inheriting from AbstractDiff.
    py::class_<AngleDiffPrecomputed, AbstractDiff>(image_labelers, "AngleDiffPrecomputed")
        .def(py::init<const cv::Mat*, const ProjectionParams*>())
        .def("DiffAt", &AngleDiffPrecomputed::DiffAt)
        .def("SatisfiesThreshold", &AngleDiffPrecomputed::SatisfiesThreshold)
        .def("Visualize", &AngleDiffPrecomputed::Visualize);

    // Bind the LineDistDiff class, inheriting from AbstractDiff.
    py::class_<LineDistDiff, AbstractDiff>(image_labelers, "LineDistDiff")
        .def(py::init<const cv::Mat*, const ProjectionParams*>())
        .def("DiffAt", &LineDistDiff::DiffAt)
        .def("SatisfiesThreshold", &LineDistDiff::SatisfiesThreshold)
        .def("Visualize", &LineDistDiff::Visualize);

    // LineDistDiffPrecomputed class
    py::class_<LineDistDiffPrecomputed, AbstractDiff>(image_labelers, "LineDistDiffPrecomputed")
        .def(py::init<const cv::Mat*, const ProjectionParams*>())
        .def("DiffAt", &LineDistDiffPrecomputed::DiffAt)
        .def("SatisfiesThreshold", &LineDistDiffPrecomputed::SatisfiesThreshold)
        .def("Visualize", &LineDistDiffPrecomputed::Visualize);

    // Bind the LineDistDiffPrecomputed class, inheriting from AbstractDiff
    py::class_<DiffFactory>(image_labelers, "DiffFactory")
        .def_static("Build", &DiffFactory::Build);
}