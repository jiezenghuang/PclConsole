#pragma once

#include "pcl/point_types.h"
#include "pcl/point_cloud.h"

#include <map>
#include <unordered_map>

struct Signature
{
	float f1 = 0.f, f2 = 0.f, f3 = 0.f, distance = 0.f;
	float alpha = 0.f;
};

template <typename T>
struct HashKey
{
	T k1, k2, k3, k4;

	HashKey(T _k1, T _k2, T _k3, T _k4)
		: k1(_k1), k2(_k2), k3(_k3), k4(_k4)
	{
	}

	bool operator==(const HashKey& h) const noexcept
	{
		return k1 == h.k1 && k2 == h.k2 && k3 == h.k3 && k4 == h.k4;
	}

	size_t operator()(const HashKey& h) const noexcept
	{
		return hash<T>()(h.k1) ^ hash<T>()(h.k2) ^ hash<T>()(h.k3) ^ hash<T>()(h.k4);
	}
};


class PointPairFeature
{
public:
	PointPairFeature(pcl::PointCloud<pcl::PointNormal>::Ptr& scene_with_normals)
		: scene_with_normals_(scene_with_normals)
	{}

	void compute_signature(pcl::PointCloud<pcl::PointNormal>::Ptr& model_with_normals);

	void match(pcl::PointCloud<pcl::PointNormal>::Ptr& model_with_normals, std::vector<Eigen::Matrix4f>& transformation, int max = 1);

	uint32_t SceneReferencePointSamplingRate;
	float PositionClusteringThreshold;
	float RotationClusteringThreshold;
	float angle_discretization_step = 12.0f / 180.0f * static_cast<float>(M_PI);
	float distance_discretization_step = 0.01f;

private:
	pcl::PointCloud<pcl::PointNormal>::Ptr scene_with_normals_;
	std::map<std::pair<std::size_t, std::size_t>, Signature> scene_feature_, model_feature_;
	std::unordered_map<HashKey<int>, std::pair<std::size_t, std::size_t>> hash_map_;
	Eigen::Matrix<Signature, Eigen::Dynamic, Eigen::Dynamic> signature_matrix_;	
	float max_distance_;

	void discretization(const float& f1, const float& f2, const float& f3, const float& f4,
		int& d1, int& d2, int& d3, int& d4);
	void compute_pair_feature(const Eigen::Vector4f& p1, const Eigen::Vector4f& n1, const Eigen::Vector4f& p2, const Eigen::Vector4f& n2,
		float& f1, float& f2, float& f3, float& f4);
	Eigen::Affine3f transform_to_x(const pcl::PointNormal& point_with_normal);
	Eigen::Affine3f transform_to_x(const Eigen::Vector3f& point, const Eigen::Vector3f& normal);
	float transform_alpha_to_y(const Eigen::Vector3f& point, const Eigen::Affine3f& transform);
};

