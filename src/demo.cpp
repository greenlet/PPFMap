#include <PPFMap/PPFMap.h>
#include <PPFMap/PPFMatch.h>
#include <PPFMap/CudaPPFMatch.h>
#include "common.h"


int main(int argc, char *argv[]) {
    char name[1024];
    const float neighborhood_radius = 0.1f;

    pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::PointCloud<pcl::PointNormal>::Ptr model_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::PointCloud<pcl::PointNormal>::Ptr scene_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::PointCloud<pcl::PointNormal>::Ptr scene_downsampled(new pcl::PointCloud<pcl::PointNormal>());
    pcl::PointCloud<pcl::PointNormal>::Ptr model_downsampled(new pcl::PointCloud<pcl::PointNormal>());

    // ========================================================================
    //  Load the point clouds of the model and the scene
    // ========================================================================

    //pcl::io::loadPCDFile("../clouds/milk.pcd", *model);
    //pcl::io::loadPCDFile("../clouds/milk_cartoon_all_small_clorox.pcd", *scene);

    pcl::io::loadPCDFile("../clouds/model_chair2.pcd", *model);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(model);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.03f);
    ne.compute(*model_normals);
    pcl::concatenateFields(*model, *model_normals, *model_with_normals);

    pcl::VoxelGrid<pcl::PointNormal> sor;
    sor.setInputCloud(model_with_normals);
    sor.setLeafSize(0.05f, 0.05f, 0.05f);
    sor.filter(*model_downsampled);

    //pcl::io::loadPCDFile("../clouds/model_chair.pcd", *model_downsampled);
    pcl::io::loadPCDFile("../clouds/scene_chair.pcd", *scene_downsampled);

    // ========================================================================
    //  Add gaussian noise to the model cloud
    // ========================================================================
    
    const float stddev = 0.0f;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    for (auto& point : *model_downsampled) {

        std::normal_distribution<float> dist_x(point.x, stddev);
        std::normal_distribution<float> dist_y(point.y, stddev);
        std::normal_distribution<float> dist_z(point.z, stddev);
        std::normal_distribution<float> dist_nx(point.normal_x, stddev);
        std::normal_distribution<float> dist_ny(point.normal_y, stddev);
        std::normal_distribution<float> dist_nz(point.normal_z, stddev);

        point.x = dist_x(generator);
        point.y = dist_y(generator);
        point.z = dist_z(generator);

        point.normal_x = dist_nx(generator);
        point.normal_y = dist_ny(generator);
        point.normal_z = dist_nz(generator);

        const float norm = point.getNormalVector3fMap().norm();

        point.normal_x /= norm;
        point.normal_y /= norm;
        point.normal_z /= norm;
    }

    // ========================================================================
    //  Transform the model cloud with a random rigid transformation.
    // ========================================================================

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*model_downsampled, centroid);

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Eigen::AngleAxisf rotation(
        Eigen::AngleAxisf(dist(generator) * static_cast<float>(M_PI), Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(dist(generator) * static_cast<float>(M_PI), Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(dist(generator) * static_cast<float>(M_PI), Eigen::Vector3f::UnitZ()));

    Eigen::Translation3f toOrigin(-centroid.head<3>());
    Eigen::Translation3f fromOrigin(centroid.head<3>());
    Eigen::Translation3f aditional(Eigen::Vector3f::Constant(-2.0f));

    Eigen::Affine3f trans(aditional * fromOrigin * rotation * toOrigin);

    pcl::transformPointCloudWithNormals(*model_downsampled, *model_downsampled, trans);
    
    // ========================================================================
    //  Compute the model's ppfs
    // ========================================================================

    // FIXME: Remove this code
    ppfmap::PPFMap<pcl::PointNormal, pcl::PointNormal> map;
    map.compute(model_downsampled, model_downsampled);
    
    int votes;

    pcl::IndicesPtr reference_point_indices(new std::vector<int>());
    for (int i = 0; i < scene_downsampled->size(); i++) {
        const auto& point = scene_downsampled->at(i);
        if (pcl::isFinite(point) && point.curvature > 0.01f) {
            reference_point_indices->push_back(i); 
        }
    }

    pcl::StopWatch timer;
    std::vector<ppfmap::Pose> poses;
    ppfmap::PPFMatch<pcl::PointNormal, pcl::PointNormal> ppf_matching;

    Eigen::Affine3f T;
    pcl::CorrespondencesPtr corr(new pcl::Correspondences());

    ppf_matching.setDiscretizationParameters(0.01f, 12.0f / 180.0f * static_cast<float>(M_PI));
    ppf_matching.setPoseClusteringThresholds(0.05f, 24.0f / 180.0f * static_cast<float>(M_PI));
    ppf_matching.setMaxRadiusPercent(0.5f);
    ppf_matching.setReferencePointIndices(reference_point_indices);

    timer.reset();
    ppf_matching.setModelCloud(model_downsampled, model_downsampled);
    std::cout << "PPF Map creation: " << timer.getTimeSeconds() << "s" <<  std::endl;

    timer.reset();
    ppf_matching.detect(scene_downsampled, scene_downsampled, T, *corr, votes);
    //ppf_matching.detect(scene_downsampled, scene_downsampled, poses);
    std::cout << "Object detection: " << timer.getTimeSeconds() << "s" <<  std::endl;


    // ========================================================================
    //  Visualize the clouds and the matching
    // ========================================================================

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());

    viewer->setCameraPosition(
        1.93776, 2.52103, -4.44761,         // Camera position.
        -0.510879, -0.502814, 0.516093,     // Focal point.
        -0.791964, 0.610274, -0.0189099);       // Up vector.

    viewer->addPointCloud<pcl::PointNormal>(model_downsampled, "model_downsampled");
    viewer->addPointCloud<pcl::PointNormal>(scene_downsampled, "scene_downsampled");

    for (const auto& c : *corr) {
        auto& scene_point = scene_downsampled->at(c.index_query);
        auto& model_point = model_downsampled->at(c.index_match);

        sprintf(name, "line_%d_%d", c.index_query, c.index_match);    
        viewer->addLine(scene_point, model_point, 1.0f, 0.0f, 0.0f, name);
    }

    //for (const auto& pose : poses) {
    
        viewer->removeShape("reference_line");
        viewer->removePointCloud("model_transformed");
        pcl::PointCloud<pcl::PointNormal>::Ptr model_transformed(new pcl::PointCloud<pcl::PointNormal>());
        pcl::transformPointCloud(*model_downsampled, *model_transformed, T);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> green(model_transformed, 1, 255, 0);
        viewer->addPointCloud<pcl::PointNormal>(model_transformed, green, "model_transformed");

        /*
         *auto& scene_point = scene_downsampled->at(pose.c.index_query);
         *auto& model_point = model_downsampled->at(pose.c.index_match);
         *viewer->addLine(scene_point, model_point, 1.0f, 0.0f, 0.0f, "reference_line");
         */

        while (!viewer->wasStopped()) {
            viewer->spinOnce();
        }
        viewer->resetStoppedFlag();
    //}

    return 0;
}
