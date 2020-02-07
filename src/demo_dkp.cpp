#include "common.h"

namespace fs = boost::filesystem;
using PCPt = pcl::PointCloud<pcl::PointXYZ>;
using PCPtPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
using PCNorm = pcl::PointCloud<pcl::Normal>;
using PCNormPtr = pcl::PointCloud<pcl::Normal>::Ptr;
using PCPtNorm = pcl::PointCloud<pcl::PointNormal>;
using PCPtNormPtr = pcl::PointCloud<pcl::PointNormal>::Ptr;


const std::string MODEL_PLY_PATH = "/home/burakov/prog/depth/research/model.ply";
const std::string MODEL_STL_PATH = "/home/burakov/prog/tra/resources/parts/front_left_node_1/model.stl";
const std::string MODEL_PCD_PATH = "/home/burakov/prog/tmp/model.pcd";

const std::string GT_PATH = "/media/burakov/HardDrive/Data/CORNER_NODE_LH_1000093_E_CORNER_NODE_RH_1000094_D_on_agv_000500";
const std::string PRED_PATH = "/home/burakov/prog/depth/bts/result_bts_tra_2";
const std::regex IMG_NAME_PAT("cam_(\\w+)_img_(\\w+).png", std::regex_constants::icase);

// const std::string GT_PATH = "/media/burakov/HardDrive/Data/20200130_fslab1";
// const std::string PRED_PATH = "/home/burakov/prog/depth/bts/result_bts_tra_2_rl";
// const std::regex IMG_NAME_PAT("cam_(\\d+)_(\\d+)(?:_None)?.png", std::regex_constants::icase);

// const std::string GT_PATH = "/media/burakov/HardDrive/Data/CORNER_NODE_LH_1000093_E_CORNER_NODE_RH_1000094_D_on_agv_000500";
// const std::string PRED_PATH = "/home/burakov/prog/depth/bts/result_bts_tra_3_agv";
// const std::regex IMG_NAME_PAT("cam_(\\w+)_img_(\\w+).png", std::regex_constants::icase);

// const std::string GT_PATH = "/media/burakov/HardDrive/Data/20200130_fslab1";
// const std::string PRED_PATH = "/home/burakov/prog/depth/bts/result_bts_tra_3_rl";
// // const std::string PRED_PATH = "/home/burakov/prog/depth/bts/result_bts_nyu_rl";
// const std::regex IMG_NAME_PAT("cam_(\\d+)_(\\d+)(?:_None)?.png", std::regex_constants::icase);

// const std::string GT_PATH = "/media/burakov/HardDrive/Data/2Parts_assembling_left_001000";
// const std::string PRED_PATH = "/home/burakov/prog/depth/bts/result_bts_tra_3_assembling";
// const std::regex IMG_NAME_PAT("cam_(\\w+)_img_(\\w+).png", std::regex_constants::icase);


pcl::PolygonMesh::Ptr read_mesh(const std::string &path, float scale = 1 / 1000.0) {
    pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
    pcl::io::loadPolygonFileSTL(MODEL_STL_PATH, *mesh);

    if (std::abs(scale - 1) > 1e-6) {
        PCPtPtr mesh_pc(new PCPt());
        pcl::fromPCLPointCloud2(mesh->cloud, *mesh_pc);
        for (pcl::PointXYZ &p : mesh_pc->points) {
            p.x *= scale;
            p.y *= scale;
            p.z *= scale;
        }
        pcl::toPCLPointCloud2(*mesh_pc, mesh->cloud);
    }

    return mesh;
}

void calc_point_normal(const pcl::PointCloud<pcl::PointXYZ> &mesh_pc, const pcl::Vertices &triangle, pcl::PointNormal &pn) {
    const pcl::PointXYZ &p1 = mesh_pc.points[triangle.vertices[0]];
    const pcl::PointXYZ &p2 = mesh_pc.points[triangle.vertices[1]];
    const pcl::PointXYZ &p3 = mesh_pc.points[triangle.vertices[2]];
    pn.x = (p1.x + p2.x + p3.x) / 3;
    pn.y = (p1.y + p2.y + p3.y) / 3;
    pn.z = (p1.z + p2.z + p3.z) / 3;
    Eigen::Vector3f v1(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
    Eigen::Vector3f v2(p3.x - p1.x, p3.y - p1.y, p3.z - p1.z);
    Eigen::Vector3f n = v1.cross(v2);
    float nn = n.norm();
    if (nn > 1e-8) {
        n /= nn;
    }
    pn.normal_x = n.x();
    pn.normal_y = n.y();
    pn.normal_z = n.z();
}

PCPtNormPtr downsample(const PCPtNormPtr &pcn, float step = 0.01) {
    PCPtNormPtr res(new PCPtNorm());
    pcl::VoxelGrid<pcl::PointNormal> sor;
    sor.setInputCloud(pcn);
    sor.setLeafSize(step, step, step);
    sor.filter(*res);
    return res;
}

PCPtNormPtr mesh_to_pc(pcl::PolygonMesh::Ptr mesh) {
    PCPtPtr mesh_pc(new PCPt());

    pcl::fromPCLPointCloud2(mesh->cloud, *mesh_pc);
    std::cout << "First mesh point: " << mesh_pc->points[0] << std::endl;

    size_t n_polys = mesh->polygons.size();
    std::cout << "Mesh points: " << mesh_pc->points.size() << ". Polygons: " << n_polys << std::endl;
    PCPtNormPtr pcn(new PCPtNorm(n_polys, 1));

    for (size_t i = 0; i < n_polys; i++) {
        calc_point_normal(*mesh_pc, mesh->polygons[i], pcn->points[i]);
    }

    return pcn;
}



void show_stl() {
    pcl::StopWatch timer;
    timer.reset();
    pcl::PolygonMesh::Ptr model_mesh = read_mesh(MODEL_STL_PATH);
    std::cout << "Reading mesh: " << timer.getTimeSeconds() << "s" << std::endl;

    std::cout << "Number of polygons: " << model_mesh->polygons.size() << std::endl;

    timer.reset();
    PCPtNormPtr model_pcn = mesh_to_pc(model_mesh);
    std::cout << "Get mesh pcn: " << timer.getTimeSeconds() << "s" << std::endl;

    model_pcn = downsample(model_pcn, 0.005);
    std::cout << "Downsampled: " << model_pcn->points.size() << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    viewer->addPolygonMesh(*model_mesh, "model_mesh");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "model_mesh");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, "model_mesh");

    viewer->addPointCloud<pcl::PointNormal>(model_pcn, "model_pc");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "model_pc");
    viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(model_pcn, model_pcn, 1, 0.05, "model_normals");

    viewer->setBackgroundColor(0.3, 0.3, 0.3);
    // viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped()){
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
    }
}

void show_pcd() {
    pcl::PointCloud<pcl::PointNormal>::Ptr model_pc(new pcl::PointCloud<pcl::PointNormal>());
    pcl::io::loadPCDFile(MODEL_PCD_PATH, *model_pc);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    viewer->setBackgroundColor(0.3, 0.3, 0.3);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color(model_pc, 0, 255, 0);
    viewer->addPointCloud<pcl::PointNormal>(model_pc, single_color, "model");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "model");
    viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(model_pc, model_pc, 10, 0.05, "model_normals");

    // viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped()){
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
    }

}

struct PtNormVisibility {
    std::string name;
    bool visible;
    PCPtNormPtr pcn;
    int viewport;
};

struct ToggleNormalsPayload {
    pcl::visualization::PCLVisualizer::Ptr viewer;
    std::vector<PtNormVisibility> pcn_vis;
};

void on_keyboard_event(const pcl::visualization::KeyboardEvent &event, void *payload_void)
{
    ToggleNormalsPayload *payload = static_cast<ToggleNormalsPayload *>(payload_void);
    if (event.getKeySym() == "n" && event.keyDown()) {
        for (auto &item : payload->pcn_vis) {
            if (item.visible) {
                payload->viewer->removePointCloud(item.name);
            } else {
                payload->viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(
                    item.pcn, item.pcn, 10, 0.05, item.name, item.viewport);
                payload->viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 0, item.name, item.viewport);
            }
            item.visible = !item.visible;
        }
    }
}

void show_stl_pcd() {
    pcl::PolygonMesh::Ptr model_mesh = read_mesh(MODEL_STL_PATH);
    PCPtNormPtr model_mesh_pc = mesh_to_pc(model_mesh);

    pcl::PointCloud<pcl::PointNormal>::Ptr model_pc(new pcl::PointCloud<pcl::PointNormal>());
    pcl::io::loadPCDFile(MODEL_PCD_PATH, *model_pc);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    std::shared_ptr<ToggleNormalsPayload> event_payload = std::make_shared<ToggleNormalsPayload>();
    event_payload->viewer = viewer;

    int v1(0), v2(1);

    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(0.3, 0.3, 0.3, v1);
    viewer->addPolygonMesh(*model_mesh, "model_mesh", v1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "model_mesh");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, "model_mesh");
    viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(model_mesh_pc, model_mesh_pc, 10, 0.05, "model_mesh_normals", v1);
    event_payload->pcn_vis.push_back({"model_mesh_normals", true, model_mesh_pc, v1});

    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor(0.0, 0.0, 0.0, v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color(model_pc, 0, 255, 0);
    viewer->addPointCloud<pcl::PointNormal>(model_pc, single_color, "model_pc", v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "model_pc");
    viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(model_pc, model_pc, 10, 0.05, "model_normals", v2);
    event_payload->pcn_vis.push_back({"model_normals", true, model_pc, v2});

    viewer->registerKeyboardCallback(on_keyboard_event, event_payload.get());

    // viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped()){
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
    }

}

using SceneCameraIds = std::vector<std::tuple<std::string, std::string, fs::path>>;

SceneCameraIds get_scene_camera_ids(const fs::path &pred_path) {
    SceneCameraIds res;

    for (const auto &entry : fs::directory_iterator(pred_path / "raw")) {
        std::smatch m;
        const std::string file_name = entry.path().filename().string();
        // std::cout << file_name << std::endl;
        if (std::regex_match(file_name, m, IMG_NAME_PAT)) {
            res.emplace_back(m[2], m[1], entry.path());
        }
    }

    return res;
}


namespace PJ = Poco::JSON;
using PJObj = PJ::Object::Ptr;
using PJArr = PJ::Array::Ptr;

static PJ::Object::Ptr get_meta(const std::string &file_path) {
	std::ifstream file_stream(file_path);
	std::string file_content((std::istreambuf_iterator<char>(file_stream)), std::istreambuf_iterator<char>());
	PJ::Parser parser;
	Poco::Dynamic::Var result = parser.parse(file_content);
	return result.extract<Poco::JSON::Object::Ptr>();
}

std::string cv_type_to_str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

std::string cv_mat_to_str(const cv::Mat &m) {
    std::ostringstream os;
    os << m.cols << "x" << m.rows << " " << cv_type_to_str(m.type());
    return os.str();
}


Eigen::Affine3f json_pose_to_transform(const PJObj &pose_json) {
    PJObj p_json = pose_json->getObject("p");
    PJObj q_json = pose_json->getObject("q");
    Eigen::Quaternionf q(q_json->getValue<float>("qw"), q_json->getValue<float>("qx"),
            q_json->getValue<float>("qy"), q_json->getValue<float>("qz"));
    Eigen::Translation3f p(p_json->getValue<float>("x"), p_json->getValue<float>("y"), p_json->getValue<float>("z"));
    Eigen::Affine3f tr = p * q * Eigen::Affine3f::Identity();
    return tr;
}


PCNormPtr calc_normals(const PCPtPtr &pc) {
    PCNormPtr res(new PCNorm());
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    // ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
    ne.setInputCloud(pc);
    ne.setMaxDepthChangeFactor(0.05);
    // ne.setRadiusSearch(0.03f);
    ne.compute(*res);
    return res;
}


PCPtNormPtr calc_pcn_from_depth(const cv::Mat &depth_cv, const PJObj &cam_json) {
    PJObj intr_json = cam_json->getObject("intrinsics");
    int w_src = intr_json->getValue<int>("w");
    int h_src = intr_json->getValue<int>("h");
    float fx = intr_json->getValue<float>("fx");
    float fy = intr_json->getValue<float>("fy");
    float cx = intr_json->getValue<float>("cx");
    float cy = intr_json->getValue<float>("cy");
    int w = depth_cv.cols;
    int h = depth_cv.rows;

    PJObj extr_json = cam_json->getObject("extrinsics");
    Eigen::Affine3f ext_tr = json_pose_to_transform(extr_json);
    // std::cout << "Camera extrinsics: ";
    // extr_json->stringify(std::cout, 4);
    // std::cout << "\nTransform: \n" << ext_tr.matrix() << std::endl;

    auto t1 = std::chrono::system_clock::now();
    Eigen::MatrixXf depth;
    cv::cv2eigen(depth_cv, depth);

    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(w, 0, w - 1);
    Eigen::VectorXf y = Eigen::VectorXf::LinSpaced(h, 0, h - 1);

    float scale = 2;
    float x_offset = (w_src - w * scale) / 2;
    float y_offset = (h_src - h * scale) / 2;
    
    x = (x.array() * scale + x_offset - cx) / fx;
    y = (y.array() * scale + y_offset - cy) / fy;

    PCPtPtr pc(new PCPt(w, h));
    for (int ih = 0; ih < h; ih++) {
        for (int iw = 0; iw < w; iw++) {
            float d = depth(ih, iw);
            auto &pt = pc->at(iw, ih);
            Eigen::Vector3f p;
            pt.x = x(iw) * d;
            pt.y = y(ih) * d;
            pt.z = d;
        }
    }

    PCNormPtr normals = calc_normals(pc);
    
    PCPtNormPtr pcn(new PCPtNorm());
    pcl::concatenateFields(*pc, *normals, *pcn);
    pcl::transformPointCloudWithNormals<pcl::PointNormal>(*pcn, *pcn, ext_tr, false);

    auto t2 = std::chrono::system_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Calculation time: " << delta.count() << " ms" << std::endl;

    return pcn;
}


pcl::PolygonMesh::Ptr transform(const pcl::PolygonMesh::Ptr &mesh, const Eigen::Affine3f &tr) {
    PCPtPtr mesh_pc(new PCPt());
    pcl::fromPCLPointCloud2(mesh->cloud, *mesh_pc);

    pcl::transformPointCloud(*mesh_pc, *mesh_pc, tr.matrix());
    pcl::PCLPointCloud2::Ptr mesh_pc2(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*mesh_pc, *mesh_pc2);

    pcl::PolygonMesh::Ptr res(new pcl::PolygonMesh());
    res->header = mesh->header;
    res->cloud = *mesh_pc2;
    res->polygons = mesh->polygons;
    
    return res;
}


std::tuple<pcl::IndicesPtr, PCPtPtr> get_reference_indices_points(const PCPtNormPtr &pcn) {
    pcl::IndicesPtr ref_inds(new std::vector<int>());

    for (int i = 0; i < pcn->size(); i++) {
        const auto& point = pcn->at(i);
        if (pcl::isFinite(point) && point.curvature > 0.01f) {
            ref_inds->push_back(i); 
        }
    }

    PCPtPtr ref_pc(new PCPt(ref_inds->size(), 1));
    for (int i = 0; i < ref_inds->size(); i++) {
        int ref_ind = ref_inds->at(i);
        ref_pc->points[i].x = pcn->points[ref_ind].x;
        ref_pc->points[i].y = pcn->points[ref_ind].y;
        ref_pc->points[i].z = pcn->points[ref_ind].z;
    }

    return std::make_pair(ref_inds, ref_pc);
}


void show_scene(PCPtNormPtr scene_pcn, PJObj meta_json, PJObj cam_json, pcl::visualization::PCLVisualizer::Ptr viewer,
        int viewport, std::shared_ptr<ToggleNormalsPayload> event_payload, bool show_gt, pcl::PolygonMeshPtr model_mesh, bool show_reference_points) {
    std::string viewport_str = std::to_string(viewport);
    auto postfix = [&viewport_str](const std::string &s) -> std::string {
        return s + "_" + viewport_str;
    };

    viewer->addPointCloud<pcl::PointNormal>(scene_pcn, postfix("scene"), viewport);

    if (show_reference_points) {
        PCPtPtr scene_ref_pc;
        std::tie(std::ignore, scene_ref_pc) = get_reference_indices_points(scene_pcn);
        viewer->addPointCloud<pcl::PointXYZ>(scene_ref_pc, postfix("scene_ref"), viewport);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, postfix("scene_ref"), viewport);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, postfix("scene_ref"), viewport);
    }

    if (show_gt && meta_json->has("parts")) {
        Eigen::Affine3f cam_pose = json_pose_to_transform(cam_json->getObject("extrinsics"));
        PJObj parts_json = meta_json->getObject("parts");
        if (parts_json->has("100")) {
            Eigen::Affine3f corner_left_pose_gt = json_pose_to_transform(parts_json->getObject("100"));
            pcl::PolygonMesh::Ptr corner_left_mesh = transform(model_mesh, corner_left_pose_gt);
            viewer->addPolygonMesh(*corner_left_mesh, postfix("corner_left_gt"), viewport);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, postfix("corner_left_gt"), viewport);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, postfix("corner_left_gt"), viewport);
        }

        if (parts_json->has("101")) {
            Eigen::Affine3f corner_right_pose_gt = json_pose_to_transform(parts_json->getObject("101"));
            // Flip right node
            corner_right_pose_gt = corner_right_pose_gt * Eigen::Quaternionf(0, 1, 0, 0);
            pcl::PolygonMesh::Ptr corner_right_mesh = transform(model_mesh, corner_right_pose_gt);
            viewer->addPolygonMesh(*corner_right_mesh, postfix("corner_right_gt"), viewport);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, postfix("corner_right_gt"), viewport);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, postfix("corner_right_gt"), viewport);
        }
    }

    std::string normals_name = postfix("scene_normals");
    event_payload->pcn_vis.push_back({normals_name, false, scene_pcn, viewport});
}


void show_scene() {
    pcl::PolygonMesh::Ptr model_mesh = read_mesh(MODEL_STL_PATH);
    fs::path gt_path(GT_PATH), pred_path(PRED_PATH);
    SceneCameraIds scene_camera_ids = get_scene_camera_ids(pred_path);
    
    const int ind = 30;
    const auto &scid = scene_camera_ids[ind];
    std::string scene_id = std::get<0>(scid);
    std::string cam_id = std::get<1>(scid);
    fs::path depth_pred_path = std::get<2>(scid);
    fs::path depth_gt_path = gt_path / "bts" / "data" / ("cam_" + cam_id + "_depth_map_" + scene_id + ".png");

    std::cout << "Camera: " << cam_id << ". Scene: " << scene_id << std::endl;
    fs::path meta_path = gt_path / "meta" / ("meta_" + scene_id + ".json");
    std::cout << "Reading meta: " << meta_path << std::endl;
    PJObj meta_json = get_meta(meta_path.string());
    PJObj cam_json = meta_json->getObject("cams")->getObject(cam_id);
    if (cam_json->has("key_points_info")) {
        cam_json->remove("key_points_info");
    }
    std::cout << "Camera " << cam_id << ": ";
    cam_json->stringify(std::cout, 4);
    std::cout << std::endl;

    std::cout << "Reading depth: " << depth_pred_path << std::endl;
    cv::Mat depth_pred = cv::imread(depth_pred_path.string(), -1);
    depth_pred.convertTo(depth_pred, CV_32FC1, 1 / 1000.0);
    // std::cout << "depth_pred: " << cv_mat_to_str(depth_pred) << std::endl;

    PCPtNormPtr scene_pred_pcn = calc_pcn_from_depth(depth_pred, cam_json);
    PCPtNormPtr scene_gt_pcn;
    if (fs::exists(depth_gt_path)) {
        cv::Mat depth_gt;
        depth_gt = cv::imread(depth_gt_path.string(), -1);
        depth_gt.convertTo(depth_gt, CV_32FC1, 1 / 1000.0);
        scene_gt_pcn = calc_pcn_from_depth(depth_gt, cam_json);
    }

    // PCPtNormPtr scene_pcn = calc_pcn_from_depth(depth_gt, cam_json);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    std::shared_ptr<ToggleNormalsPayload> event_payload = std::make_shared<ToggleNormalsPayload>();
    event_payload->viewer = viewer;

    int v1(0), v2(1);
    if (!scene_gt_pcn) {
        viewer->createViewPort(0, 0, 1, 1, v1);
        viewer->setBackgroundColor(0.3, 0.3, 0.3, v1);
        show_scene(scene_pred_pcn, meta_json, cam_json, viewer, v1, event_payload, true, model_mesh, false);
    } else {
        viewer->createViewPort(0, 0, 0.5, 1, v1);
        viewer->setBackgroundColor(0.3, 0.3, 0.3, v1);
        viewer->createViewPort(0.5, 0, 1, 1, v2);
        viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
        show_scene(scene_pred_pcn, meta_json, cam_json, viewer, v1, event_payload, true, model_mesh, false);
        show_scene(scene_gt_pcn, meta_json, cam_json, viewer, v2, event_payload, true, model_mesh, false);
    }

    viewer->registerKeyboardCallback(on_keyboard_event, event_payload.get());

    Eigen::Affine3f cam_pose = json_pose_to_transform(cam_json->getObject("extrinsics"));

    viewer->initCameraParameters();
    Eigen::Vector3f cam_pose_tr = cam_pose.translation();
    Eigen::AngleAxisf cam_pose_rot(cam_pose.rotation());
    Eigen::Vector3f cam_view_dir = cam_pose_rot * Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f cam_up_dir = cam_pose_rot * Eigen::Vector3f(0, -1, 0);
    viewer->setCameraPosition(cam_pose_tr.x(), cam_pose_tr.y(), cam_pose_tr.z(),
            cam_view_dir.x(), cam_view_dir.y(), cam_view_dir.z(),
            cam_up_dir.x(), cam_up_dir.y(), cam_up_dir.z());

    viewer->addCoordinateSystem(1.0);
    while (!viewer->wasStopped()){
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
    }

}


std::pair<std::vector<ppfmap::Pose>, PCPtPtr> ppf_match(const PCPtNormPtr &scene_pcn, const PCPtNormPtr &model_pcn) {
    pcl::IndicesPtr scene_ref_inds;
    PCPtPtr scene_ref_pc;
    std::tie(scene_ref_inds, scene_ref_pc) = get_reference_indices_points(scene_pcn);
    std::cout << "Total scene reference points: " << scene_ref_inds->size() << std::endl;

    pcl::StopWatch timer;
    std::vector<ppfmap::Pose> poses;
    ppfmap::PPFMatch<pcl::PointNormal, pcl::PointNormal> ppf_matching;

    ppf_matching.setDiscretizationParameters(0.03f, 12.0f / 180.0f * static_cast<float>(M_PI));
    ppf_matching.setPoseClusteringThresholds(0.25f, 24.0f / 180.0f * static_cast<float>(M_PI));
    ppf_matching.setMaxRadiusPercent(0.5f);
    ppf_matching.setReferencePointIndices(scene_ref_inds);

    timer.reset();
    ppf_matching.setModelCloud(model_pcn, model_pcn);
    std::cout << "PPF Map creation: " << timer.getTimeSeconds() << "s" <<  std::endl;

    timer.reset();
    ppf_matching.detect(scene_pcn, scene_pcn, poses);
    std::cout << "Object detection: " << timer.getTimeSeconds() << "s" <<  std::endl;

    return std::make_pair(poses, scene_ref_pc);
}


void run_ppf() {
    pcl::PolygonMesh::Ptr model_mesh = read_mesh(MODEL_STL_PATH);
    PCPtNormPtr model_pcn = mesh_to_pc(model_mesh);

    // PCPtNormPtr model_pcn(new PCPtNorm());
    // pcl::io::loadPCDFile(MODEL_PCD_PATH, *model_pcn);

    std::cout << "Model point cloud size: " << model_pcn->points.size() << std::endl;

    fs::path gt_path(GT_PATH), pred_path(PRED_PATH);
    SceneCameraIds scene_camera_ids = get_scene_camera_ids(pred_path);
    const int ind = 30;
    const auto &scid = scene_camera_ids[ind];
    std::string scene_id = std::get<0>(scid);
    std::string cam_id = std::get<1>(scid);
    fs::path depth_pred_path = std::get<2>(scid);
    fs::path depth_gt_path = gt_path / "bts" / "data" / ("cam_" + cam_id + "_depth_map_" + scene_id + ".png");

    std::cout << "Camera: " << cam_id << ". Scene: " << scene_id << std::endl;
    fs::path meta_path = gt_path / "meta" / ("meta_" + scene_id + ".json");
    std::cout << "Reading meta: " << meta_path << std::endl;
    PJObj meta_json = get_meta(meta_path.string());
    PJObj cam_json = meta_json->getObject("cams")->getObject(cam_id);
    if (cam_json->has("key_points_info")) {
        cam_json->remove("key_points_info");
    }
    std::cout << "Camera " << cam_id << ": ";
    cam_json->stringify(std::cout, 4);
    std::cout << std::endl;

    std::cout << "Reading depth: " << depth_pred_path << std::endl;
    cv::Mat depth_pred = cv::imread(depth_pred_path.string(), -1);
    depth_pred.convertTo(depth_pred, CV_32FC1, 1 / 1000.0);
    // std::cout << "depth_pred: " << cv_mat_to_str(depth_pred) << std::endl;

    cv::Mat depth_gt;
    if (fs::exists(depth_gt_path)) {
        depth_gt = cv::imread(depth_gt_path.string(), -1);
        depth_gt.convertTo(depth_gt, CV_32FC1, 1 / 1000.0);
    }

    PCPtNormPtr scene_pcn = calc_pcn_from_depth(depth_gt, cam_json);
    // PCPtNormPtr scene_pcn = calc_pcn_from_depth(depth_pred, cam_json);

    PCPtNormPtr model_pcn_down = downsample(model_pcn, 0.03);
    PCPtNormPtr scene_pcn_down = downsample(scene_pcn, 0.03);


    std::vector<ppfmap::Pose> poses;
    PCPtPtr scene_ref_pc;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    viewer->setBackgroundColor(0.3, 0.3, 0.3);

    viewer->addPointCloud<pcl::PointNormal>(scene_pcn, "scene");

    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color(model_pcn_down, 0, 255, 0);
    // viewer->addPointCloud<pcl::PointNormal>(model_pcn_down, single_color, "model_pc");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "model_pc");
    // viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(model_pcn_down, model_pcn_down, 10, 0.05, "model_normals");    

    std::tie(poses, scene_ref_pc) = ppf_match(scene_pcn_down, model_pcn_down);
    // std::tie(poses, scene_ref_pc) = ppf_match(scene_pcn_down, model_pcn);
    // std::tie(poses, scene_ref_pc) = ppf_match(scene_pcn, model_pcn);
    
    viewer->addPointCloud<pcl::PointXYZ>(scene_ref_pc, "scene_ref");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scene_ref");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "scene_ref");

    std::cout << poses.size() << " poses detected" << std::endl;
    int max_poses = 2;
    int min_votes = 20;
    ppfmap::Pose last_pose;
    for (int i = 0, i_pose = 0; i < poses.size() && i_pose < max_poses; i++) {
        const ppfmap::Pose &pose = poses[i];
        if (i_pose == 0 || (pose.t.translation() - last_pose.t.translation()).norm() > 0.15) {
            last_pose = pose;
            i_pose++;
            // std::cout << " ---- Pose votes: " << pose.votes << ". T:\n" << pose.t.matrix() << std::endl;
            Eigen::Affine3f T = pose.t;
            pcl::PolygonMesh::Ptr model_mesh_pred = transform(model_mesh, T);
            std::string model_id = "model_pred" + std::to_string(i_pose);
            // std::cout << model_id << std::endl;
            viewer->addPolygonMesh(*model_mesh_pred, model_id);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, model_id);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, model_id);
        }
        if (pose.votes < min_votes) {
            break;
        }
    }

    Eigen::Affine3f cam_pose = json_pose_to_transform(cam_json->getObject("extrinsics"));

    // if (meta_json->has("parts")) {
    //     PJObj parts_json = meta_json->getObject("parts");
    //     Eigen::Affine3f corner_left_pose_gt = json_pose_to_transform(parts_json->getObject("100"));
    //     Eigen::Affine3f corner_right_pose_gt = json_pose_to_transform(parts_json->getObject("101"));
    //     // Flip right node
    //     corner_right_pose_gt = corner_right_pose_gt * Eigen::Quaternionf(0, 1, 0, 0);
    //     pcl::PolygonMesh::Ptr corner_left_mesh = transform(model_mesh, corner_left_pose_gt);
    //     pcl::PolygonMesh::Ptr corner_right_mesh = transform(model_mesh, corner_right_pose_gt);

    //     // viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(scene_pcn, scene_pcn, 20, 0.05, "scene_normals");

    //     // viewer->addPolygonMesh(*corner_left_mesh, "corner_left_gt");
    //     // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "corner_left_gt");
    //     // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, "corner_left_gt");
    //     // viewer->addPolygonMesh(*corner_right_mesh, "corner_right_gt");
    //     // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "corner_right_gt");
    //     // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, "corner_right_gt");
    // }

    viewer->initCameraParameters();
    Eigen::Vector3f cam_pose_tr = cam_pose.translation();
    Eigen::AngleAxisf cam_pose_rot(cam_pose.rotation());
    Eigen::Vector3f cam_view_dir = cam_pose_rot * Eigen::Vector3f(0, 0, 1);
    Eigen::Vector3f cam_up_dir = cam_pose_rot * Eigen::Vector3f(0, -1, 0);
    viewer->setCameraPosition(cam_pose_tr.x(), cam_pose_tr.y(), cam_pose_tr.z(),
            cam_view_dir.x(), cam_view_dir.y(), cam_view_dir.z(),
            cam_up_dir.x(), cam_up_dir.y(), cam_up_dir.z());

    viewer->addCoordinateSystem(1.0);
    while (!viewer->wasStopped()){
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
    }

}



int main(int argc, char *argv[]) {
    std::cout << "GT path: " << GT_PATH << std::endl;
    std::cout << "Pred path: " << PRED_PATH << std::endl;

    // show_stl();
    // show_pcd();
    // show_stl_pcd();
    // show_scene();
    run_ppf();

    return 0;
}

