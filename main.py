import open3d as o3d
import numpy as np


# Example 1 - Original Stanford Bunny Mesh
bunny = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(bunny.path)

# Prettify the mesh for visualization.
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.5, 0.5, 0.5])

o3d.visualization.draw_geometries([mesh])


# Example 2 - Sample a Point Cloud
def get_bunny_point_cloud(
    number_of_points: int = 5000, visualize: bool = False
) -> o3d.geometry.PointCloud:
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)

    mesh.compute_vertex_normals()

    pc = mesh.sample_points_poisson_disk(
        number_of_points=number_of_points, init_factor=5
    )

    if visualize:
        o3d.visualization.draw_geometries([pc])

    return pc


bunny_pc = get_bunny_point_cloud(visualize=True)


# Example 3 - Ball Pivoting
def build_mesh_with_ball_pivoting(
    pc: o3d.geometry.PointCloud, visualize: bool = False
) -> o3d.geometry.TriangleMesh:
    radii = [0.005, 0.01, 0.02, 0.04]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pc, o3d.utility.DoubleVector(radii)
    )

    # Prettify the mesh for visualization.
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])

    if visualize:
        o3d.visualization.draw_geometries([mesh])

    return mesh


ball_pivoting_mesh = build_mesh_with_ball_pivoting(bunny_pc, visualize=True)


# Example 4 - Ball Pivoting on Cropped Point Cloud
aabb = bunny_pc.get_axis_aligned_bounding_box()

max_bound = aabb.get_max_bound()
min_bound = aabb.get_min_bound()
difference = max_bound - min_bound

new_max_bound = np.array(
    [
        max_bound[0],
        max_bound[1] - difference[1] * 0.2,
        max_bound[2],
    ]
)
new_min_bound = np.array(
    [
        min_bound[0],
        min_bound[1] + difference[1] * 0.2,
        min_bound[2],
    ]
)
shrunk_aabb = o3d.geometry.AxisAlignedBoundingBox(new_min_bound, new_max_bound)

cropped_bunny_pc = bunny_pc.crop(shrunk_aabb)

build_mesh_with_ball_pivoting(cropped_bunny_pc, visualize=True)


# Example 5 - Poisson Surface Reconstruction
def build_mesh_with_poisson(
    pc: o3d.geometry.PointCloud, visualize: bool = False
) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=9
    )

    mesh.compute_vertex_normals()  # For shading
    mesh.paint_uniform_color([0.5, 0.5, 0.5])

    if visualize:
        o3d.visualization.draw_geometries([mesh])

    return mesh


build_mesh_with_poisson(bunny_pc, visualize=True)


# Example 6 - Poisson Surface Reconstruction with Cropped Point Cloud
build_mesh_with_poisson(cropped_bunny_pc, visualize=True)


# Example 7 - Spherical Projection
def spherify(
    pc: o3d.geometry.PointCloud,
    center: np.ndarray = np.zeros(3),
    radius: float = 1.0,
    visualize: bool = False,
) -> o3d.geometry.PointCloud:
    points = np.asarray(pc.points)
    colors = pc.colors

    # Compute distance from center for all points.
    distances = np.linalg.norm(points - center, axis=1)

    # Normalize all points so that distance from the center is
    # equal to the radius.
    spherified_points = (points.T * (radius / distances)).T

    spherified_pc = o3d.geometry.PointCloud()
    spherified_pc.points = o3d.utility.Vector3dVector(spherified_points)
    spherified_pc.colors = colors

    if visualize:
        o3d.visualization.draw_geometries([spherified_pc])

    return spherified_pc


center = bunny_pc.get_center()

# Invert the center so that it's subtracted from all of the points
# in order to move the origin to the center of the point cloud.
translated_bunny_pc = bunny_pc.translate(-center)

bunny_sphere_pc = spherify(translated_bunny_pc, radius=1, visualize=True)


# Example 8 - Sphere to Mesh
def build_mesh_from_spherified_points(
    original_pc: o3d.geometry.PointCloud,
    spherified_points: np.ndarray,
    visualize: bool = False,
) -> o3d.geometry.TriangleMesh:
    """
    Assumes that the points in the original point cloud and
    the spherified points are in the same order.
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(spherified_points)

    # Since we've projected onto a unit sphere, this will always
    # contain all of the points.
    convex_hull_mesh, convex_hull_included_indexes = pc.compute_convex_hull()

    # Each row contains the three indexes of vertices to be connected.
    triangles = np.asarray(convex_hull_mesh.triangles)

    original_points = np.asarray(original_pc.points)

    final_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(
            original_points[convex_hull_included_indexes]
        ),
        triangles=o3d.utility.Vector3iVector(triangles),
    )

    final_mesh.compute_vertex_normals()  # For shading
    final_mesh.paint_uniform_color([0.5, 0.5, 0.5])

    if visualize:
        o3d.visualization.draw_geometries([final_mesh])

    return final_mesh


# Copying to a new array fixes a strange error that I was having
# on the macOS build of Open3D.
build_mesh_from_spherified_points(
    bunny_pc, np.array(bunny_sphere_pc.points), visualize=True
)


# Example 9 - Modified Spherify
def spherify_array(
    points: np.ndarray, center: np.ndarray = np.zeros(3), radius: float = 1.0
) -> np.ndarray:
    """
    Expects points in an array with shape (n, 3).
    """
    # Compute distance from center for all points.
    distances = np.linalg.norm(points - center, axis=1)

    # Normalize all points so that distance from the center is
    # equal to the radius.
    spherified_points = (points.T * (radius / distances)).T

    return spherified_points


# Example 10 - Precompute Neighbors
def build_neighboring_index_array(
    pc: o3d.geometry.PointCloud, num_neighbors: int = 10
) -> np.ndarray:
    k = num_neighbors + 1  # knn first returns the point, itself.

    # Get the neighbors for each vertex.
    tree = o3d.geometry.KDTreeFlann(pc)
    neighboring_index_list = []
    points = np.asarray(pc.points)

    for point in points:
        # Be aware that this is incompatible with numpy 2. Reducing
        # to 1.26.4 fixed a segfault issue.
        _, indexes, _ = tree.search_knn_vector_3d(point, k)
        neighboring_index_list.append(indexes[1:])

    return np.array(neighboring_index_list)


# Example 11 - A Better Spherical Embedding
def find_spherical_embedding(
    points: np.ndarray,
    neighboring_indexes: np.ndarray,
    num_iterations: int = 1000,
    damping_coefficient: float = 0.5,
) -> np.ndarray:
    number_of_points = len(points)

    # Make a copy of the points to work with.
    working_points = np.array(points)

    for _ in range(num_iterations):
        # Move the center of the points to the origin.
        center = np.average(working_points, axis=0)
        translated_points = working_points - center

        # Project the points onto the unit sphere.
        spherified_points = spherify_array(translated_points)

        # This effectively replaces each index in the neighboring
        # indexes list with the actual point at that index.
        neighboring_points = spherified_points[neighboring_indexes]

        # For each neighboring point u_j of u_j, compute u_j - u_i.
        neighboring_differences = neighboring_points - spherified_points.reshape(
            number_of_points, 1, 3
        )

        # Get the weighted sum of the differences.
        laplacians = np.average(neighboring_differences, axis=1)

        # Compute the dot product of each laplacian with its
        # corresponding point u_i.
        dots = np.sum(laplacians * spherified_points, axis=1)

        # Compute the Laplace-Beltrami operators for all points.
        lbs = laplacians - (spherified_points.T * dots).T

        # Perturb each point based on its Laplace-Beltrami operator.
        perturbed_points = spherified_points + damping_coefficient * lbs

        # Get ready to go around again. The paper includes another
        # normalization step, but that isn't really necessary.
        working_points = perturbed_points

    return working_points


# Example 12 - Putting Everything Together
def spherically_meshify(
    pc: o3d.geometry.PointCloud,
    num_iterations: int = 1000,
    num_neighbors: int = 10,
    damping_coefficient: float = 0.5,
    visualize: bool = False,
) -> o3d.geometry.TriangleMesh:
    neighboring_indexes = build_neighboring_index_array(pc, num_neighbors)

    spherical_embedding_points = find_spherical_embedding(
        np.array(pc.points),
        neighboring_indexes,
        num_iterations=num_iterations,
        damping_coefficient=damping_coefficient,
    )

    final_mesh = build_mesh_from_spherified_points(
        pc, spherical_embedding_points, visualize=visualize
    )

    return final_mesh


for n in [1, 10, 100, 1000]:
    spherically_meshify(
        bunny_pc,
        num_iterations=n,
        num_neighbors=25,
        visualize=True,
    )

# Example 13 - More Iterations
mesh = spherically_meshify(
    bunny_pc,
    num_iterations=10_000,
    num_neighbors=25,
    visualize=True,
)

print(f"Orientable: {mesh.is_orientable()}")
print(f"Self-intersecting: {mesh.is_self_intersecting()}")
print(f"Edge Manifold: {mesh.is_edge_manifold()}")
print(f"Vertex Manifold: {mesh.is_vertex_manifold()}")
print(f"Watertight: {mesh.is_watertight()}")

# Example 14 - Running on the Cropped Bunny
mesh_from_cropped = spherically_meshify(
    cropped_bunny_pc, num_iterations=10_000, num_neighbors=25, visualize=True
)
