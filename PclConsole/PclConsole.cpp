// PclConsole.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>
#include <thread>

#include "pcl/io/pcd_io.h"
#include "pcl/features/normal_3d.h"
#include "pcl/features/ppf.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/registration/ppf_registration.h"
#include "pcl/sample_consensus/method_types.h"
#include "pcl/sample_consensus/model_types.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/visualization/pcl_visualizer.h"

using namespace std;
using namespace pcl;

const char* viewer_name = "Point Cloud Viewer";
const Eigen::Vector4f subsampling_leaf_size(2.0f, 2.0f, 2.0f, 0.0f);
const float normal_estimation_search_radius = 1.0f;

PointCloud<PointNormal>::Ptr
subsampleAndCalculateNormals(const PointCloud<PointXYZ>::Ptr& cloud)
{
    PointCloud<PointXYZ>::Ptr cloud_subsampled(new PointCloud<PointXYZ>());
    VoxelGrid<PointXYZ> subsampling_filter;
    subsampling_filter.setInputCloud(cloud);
    subsampling_filter.setLeafSize(subsampling_leaf_size);
    subsampling_filter.filter(*cloud_subsampled);

    PointCloud<Normal>::Ptr cloud_subsampled_normals(new PointCloud<Normal>());
    NormalEstimation<PointXYZ, Normal> normal_estimation_filter;
    normal_estimation_filter.setInputCloud(cloud_subsampled);
    search::KdTree<PointXYZ>::Ptr search_tree(new search::KdTree<PointXYZ>);
    normal_estimation_filter.setSearchMethod(search_tree);
    normal_estimation_filter.setRadiusSearch(normal_estimation_search_radius);
    normal_estimation_filter.compute(*cloud_subsampled_normals);

    PointCloud<PointNormal>::Ptr cloud_subsampled_with_normals(
        new PointCloud<PointNormal>());
    concatenateFields(
        *cloud_subsampled, *cloud_subsampled_normals, *cloud_subsampled_with_normals);

    PCL_INFO("Cloud dimensions before / after subsampling: %u / %u\n",
        cloud->points.size(),
        cloud_subsampled->points.size());
    return cloud_subsampled_with_normals;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        PCL_ERROR("Usage: %s scene.pcd model.txt", argv[0]);
        return -1;
    }

    PCDReader reader;
    PointCloud<PointXYZ>::Ptr scene(new PointCloud<PointXYZ>());
    reader.read(argv[1], *scene);
    PCL_INFO("read scene %s\n", argv[1]);

    std::vector<PointCloud<PointXYZ>::Ptr> model_list;
    ifstream file_stream(argv[2]);
    std::string model_file;
    while (getline(file_stream, model_file))
    {
        if (model_file.front() == '#')
            continue;

        PointCloud<PointXYZ>::Ptr model(new PointCloud<PointXYZ>());
        reader.read(model_file, *model);
        model_list.push_back(model);
        PCL_INFO("read model %s\n", model_file.c_str());
    }
    
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(1.0f);
    extract.setNegative(true);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    unsigned nr_points = unsigned(scene->points.size());
    while (scene->points.size() > 0.3 * nr_points) {
        seg.setInputCloud(scene);
        seg.segment(*inliers, *coefficients);
        PCL_INFO("Plane inliers: %u\n", inliers->indices.size());
        if (inliers->indices.size() < 50000)
          break;

        extract.setInputCloud(scene);
        extract.setIndices(inliers);
        extract.filter(*scene);
    }

    PointCloud<PointNormal>::Ptr cloud_scene_input =
        subsampleAndCalculateNormals(scene);
    std::vector<PointCloud<PointNormal>::Ptr> cloud_models_with_normals;

    PCL_INFO("Training models ...\n");
    std::vector<PPFHashMapSearch::Ptr> hashmap_search_vector;
    for (const auto& cloud_model : model_list) {
        PointCloud<PointNormal>::Ptr cloud_model_input =
            subsampleAndCalculateNormals(cloud_model);
        cloud_models_with_normals.push_back(cloud_model_input);

        PointCloud<PPFSignature>::Ptr cloud_model_ppf(new PointCloud<PPFSignature>());
        PPFEstimation<PointNormal, PointNormal, PPFSignature> ppf_estimator;
        ppf_estimator.setInputCloud(cloud_model_input);
        ppf_estimator.setInputNormals(cloud_model_input);
        ppf_estimator.compute(*cloud_model_ppf);

        //for (PPFSignature signature : cloud_model_ppf->points)
        //{
        //    signature.f1 = acosf(signature.f1);
        //    signature.f2 = acosf(signature.f2);
        //    signature.f3 = acosf(signature.f3);
        //}            

        PPFHashMapSearch::Ptr hashmap_search(
            new PPFHashMapSearch(12.0f / 180.0f * float(M_PI), 0.05f));
        hashmap_search->setInputFeatureCloud(cloud_model_ppf);
        hashmap_search_vector.push_back(hashmap_search);
    }

    visualization::PCLVisualizer viewer(viewer_name);
    viewer.setBackgroundColor(0, 0, 0);
    //viewer.addPointCloud(scene);
    //viewer.spinOnce(10);
    PCL_INFO("Registering models to scene ...\n");
    for (std::size_t model_i = 0; model_i < model_list.size(); ++model_i) {

        PPFRegistration<PointNormal, PointNormal> ppf_registration;
        // set parameters for the PPF registration procedure
        ppf_registration.setSceneReferencePointSamplingRate(100);
        ppf_registration.setPositionClusteringThreshold(1.0f);
        ppf_registration.setRotationClusteringThreshold(24.0f / 180.0f * float(M_PI));
        ppf_registration.setSearchMethod(hashmap_search_vector[model_i]);
        ppf_registration.setInputSource(cloud_models_with_normals[model_i]);
        ppf_registration.setInputTarget(cloud_scene_input);

        PointCloud<PointNormal> cloud_output_subsampled;
        ppf_registration.align(cloud_output_subsampled);
                
        PointCloud<PointXYZ>::Ptr cloud_output_subsampled_xyz(new PointCloud<PointXYZ>());
        for (const auto& point : cloud_output_subsampled.points)
            cloud_output_subsampled_xyz->points.emplace_back(point.x, point.y, point.z);

        Eigen::Matrix4f mat = ppf_registration.getFinalTransformation();
        
        Eigen::Affine3f final_transformation(mat);

        PointCloud<PointXYZ>::Ptr cloud_output(new PointCloud<PointXYZ>());
        pcl::transformPointCloud(
            *model_list[model_i], *cloud_output, final_transformation);

        stringstream ss;
        ss << "model_" << model_i;
        visualization::PointCloudColorHandlerRandom<PointXYZ> random_color(cloud_output_subsampled_xyz);
            //cloud_output->makeShared());
        viewer.addPointCloud(cloud_output_subsampled_xyz, random_color, ss.str().c_str());
        //viewer.addPointCloud(model_list[model_i]);
        PCL_INFO("Showing model %s\n", ss.str().c_str());
    }

    //viewer.addPointCloud(model_list[0]);

    PCL_INFO("All models have been registered!\n");

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
        std::this_thread::sleep_for(100ms);
    }

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
