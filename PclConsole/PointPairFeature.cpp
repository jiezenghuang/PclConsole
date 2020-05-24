#include "PointPairFeature.h"

#include "pcl/kdtree/kdtree_flann.h"


void PointPairFeature::compute_signature(pcl::PointCloud<pcl::PointNormal>::Ptr& model_with_normals)
{
	std::size_t total_points = model_with_normals->points.size();
	signature_matrix_.resize(total_points, total_points);
	for (std::size_t i = 0; i < total_points; ++i)
	{
		for (std::size_t j = 0; j < total_points; ++j)
		{
			Signature signature;
			if (i != j)
			{
				Eigen::Vector4f point_i = model_with_normals->points[i].getVector4fMap(),
					point_j = model_with_normals->points[j].getVector4fMap(),
					normal_i = model_with_normals->points[i].getNormalVector4fMap(),
					normal_j = model_with_normals->points[j].getNormalVector4fMap();

				compute_pair_feature(point_i, normal_i, point_j, normal_j, signature.f1, signature.f2, signature.f3, signature.distance);
				Eigen::Affine3f transform = transform_to_x(model_with_normals->points[i]);
				signature.alpha = transform_alpha_to_y(point_j, transform);
			}
			else
			{
				signature.f1 = signature.f2 = signature.f3 = signature.distance = signature.alpha = std::numeric_limits<float>::quiet_NaN();
			}
			signature_matrix_(i, j) = signature;
		}
	}

	hash_map_.clear();
	int d1, d2, d3, d4;
	max_distance_ = 0.f;
	for (std::size_t i = 0; i < total_points; ++i)
	{
		for (std::size_t j = 0; j < total_points; ++j)
		{
			discretization(signature_matrix_(i, j).f1, signature_matrix_(i, j).f2, signature_matrix_(i, j).f3, signature_matrix_(i, j).distance,
				d1, d2, d3, d4);
			hash_map_[HashKey<int>(d4, d1, d2, d3)] = std::pair<std::size_t, std::size_t>(i, j);

			if (max_distance_ < signature_matrix_(i, j).distance)
				max_distance_ = signature_matrix_(i, j).distance;
		}
	}
}

void PointPairFeature::match(pcl::PointCloud<pcl::PointNormal>::Ptr& model_with_normals, std::vector<Eigen::Matrix4f>& transformation, int max = 1)
{
	pcl::KdTreeFLANN<pcl::PointNormal>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointNormal>());
	tree->setInputCloud(scene_with_normals_);
	//std::map<std::pair<std::size_t, std::size_t>, Signature> 

	for (std::size_t i = 0; i < scene_with_normals_->points.size(); ++i)
	{
		Eigen::Vector3f point_i = scene_with_normals_->points[i].getVector3fMap(),
			normal_i = scene_with_normals_->points[i].getNormalVector3fMap();
		Eigen::Affine3f transform = transform_to_x(point_i, normal_i);

		std::vector<int> indices;
		std::vector<float> distances;
		tree->radiusSearch(i, max_distance_ / 2, indices, distances);
		for (std::size_t j : indices)
		{
			if (i != j)
			{
				Signature signature;
				int d1, d2, d3, d4;
				compute_pair_feature(scene_with_normals_->points[i].getVector4fMap(), scene_with_normals_->points[i].getNormalVector4fMap(),
					scene_with_normals_->points[j].getVector4fMap(), scene_with_normals_->points[j].getNormalVector4fMap(),
					signature.f1, signature.f2, signature.f3, signature.distance);
				std::vector<std::pair<std::size_t, std::size_t> > nearest_indices;
				discretization(signature.f1, signature.f2, signature.f3, signature.distance, d1, d2, d3, d4);
				auto range_pair = hash_map_.equal_range(HashKey<int>(d1, d2, d3, d4));
				for (; range_pair.first != range_pair.second; ++range_pair.first)
				{
					nearest_indices.emplace_back(range_pair.second->first, range_pair.second->second);
				}
				float alpha = transform_alpha_to_y(scene_with_normals_->points[j].getVector3fMap(), transform);
				for (auto indice_index : nearest_indices)
				{
					alpha = signature_matrix_(indice_index.first, indice_index.second).alpha - alpha;
				}
			}
		}
	}
	

	float angle_discretization_step = 12.0f / 180.0f * static_cast<float>(M_PI);
	float distance_discretization_step = 0.01f;
}

void PointPairFeature::discretization(const float& f1, const float& f2, const float& f3, const float& f4, int& d1, int& d2, int& d3, int& d4)
{
	d1 = static_cast<int> (std::floor(f1 / angle_discretization_step));
	d2 = static_cast<int> (std::floor(f2 / angle_discretization_step));
	d3 = static_cast<int> (std::floor(f3 / angle_discretization_step));
	d4 = static_cast<int> (std::floor(f4 / distance_discretization_step));
}

void PointPairFeature::compute_pair_feature(const Eigen::Vector4f& p1, const Eigen::Vector4f& n1,
	const Eigen::Vector4f& p2, const Eigen::Vector4f& n2, float& f1, float& f2, float& f3, float& f4)
{
	Eigen::Vector4f delta = p2 - p1;
	delta[3] = 0.f;
	f4 = delta.norm();

	delta = delta.normalized();
	f1 = std::acosf(n1.dot(delta));
	f2 = std::acosf(n2.dot(delta));
	f3 = std::acosf(n1.dot(n2));
}

Eigen::Affine3f PointPairFeature::transform_to_x(const pcl::PointNormal& point_with_normal)
{
	return transform_to_x(point_with_normal.getVector3fMap(), point_with_normal.getNormalVector3fMap());
}

Eigen::Affine3f PointPairFeature::transform_to_x(const Eigen::Vector3f& point, const Eigen::Vector3f& normal)
{
	float angle = std::acosf(normal.dot(Eigen::Vector3f::UnitX()));
	Eigen::Vector3f axis = (normal.y() == 0.f && normal.z() == 0.f) ?
		Eigen::Vector3f::UnitY() : normal.cross(Eigen::Vector3f::UnitX()).normalized();
	Eigen::AngleAxisf rotation(angle, axis);
	return Eigen::Affine3f(Eigen::Translation3f(rotation * ((-1) * point)) * rotation);
}

float PointPairFeature::transform_alpha_to_y(const Eigen::Vector3f& point, const Eigen::Affine3f& transform)
{
	Eigen::Vector3f transform_point = transform * point;
	float alpha = std::atan2(-transform_point(2), transform_point(1));
	if (std::sin(alpha) * transform_point(2) < 0.0f)
		alpha *= (-1);
	return -alpha;
}
