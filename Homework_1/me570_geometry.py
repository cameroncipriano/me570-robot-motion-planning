"""
Classes and functions for Polygons and Edges
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def no_edge_collisions(polygon, edge_to_check):
    edge_shape = []
    num_cols = polygon.vertices.shape[1]

    for i in range(num_cols):
        new_edge = np.vstack(
            (polygon.vertices[:, i], polygon.vertices[:, (i+1) % num_cols])).T
        edge_shape.append(Edge(new_edge))

    for edge in edge_shape:
        if edge.is_collision(edge_to_check):
            return False

    return True


def between_segment(p1, p2, p3):
    p1_x = p1[0, 0]
    p1_y = p1[1, 0]

    p2_x = p2[0, 0]
    p2_y = p2[1, 0]

    p3_x = p3[0, 0]
    p3_y = p3[1, 0]

    return (p2_x >= min(p1_x, p3_x) and (p2_x <= max(p1_x, p3_x)) and
            (p2_y >= min(p1_y, p3_y) and p2_y <= max(p1_y, p3_y)))


def angle(vertex0, vertex1, vertex2, angle_type='signed'):
    """
    Compute the angle between two edges  vertex0-vertex1 and  vertex0-vertex2 having an endpoint in
    common. The angle is computed by starting from the edge  vertex0-- vertex1, and then
    ``walking'' in a counterclockwise manner until the edge  vertex0-vertex2 is found.
    The angle is computed by starting from the vertex0-vertex1 edge, and then “walking” in a
    counterclockwise manner until the is found.
    """
    # tolerance to check for coincident points
    tol = 2.22e-16

    # compute vectors corresponding to the two edges, and normalize
    vec1 = vertex1 - vertex0
    vec2 = vertex2 - vertex0

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 < tol or norm_vec2 < tol:
        # vertex1 or vertex2 coincides with vertex0, abort
        edge_angle = math.nan
        return edge_angle

    vec1 = vec1 / norm_vec1
    vec2 = vec2 / norm_vec2

    # Transform vec1 and vec2 into flat 3-D vectors,
    # so that they can be used with np.inner and np.cross
    vec1flat = np.vstack([vec1, 0]).flatten()
    vec2flat = np.vstack([vec2, 0]).flatten()

    c_angle = np.inner(vec1flat, vec2flat)
    s_angle = np.inner(np.array([0, 0, 1]), np.cross(vec1flat, vec2flat))

    edge_angle = math.atan2(s_angle, c_angle)

    angle_type = angle_type.lower()
    if angle_type == 'signed':
        # nothing to do
        pass
    elif angle_type == 'unsigned':
        edge_angle = math.modf(edge_angle + 2 * math.pi, 2 * math.pi)
    else:
        raise ValueError('Invalid argument angle_type')

    return edge_angle


class Polygon:
    """ Class for plotting, drawing, checking visibility and collision with polygons. """

    def __init__(self, vertices):
        """
        Save the input coordinates to the internal attribute  vertices.
        """
        self.vertices = vertices

    def flip(self):
        """
        Reverse the order of the vertices (i.e., transform the polygon from filled in
        to hollow and viceversa).
        """
        self.vertices = np.fliplr(self.vertices)

    def plot(self, style):
        """
        Plot the polygon using Matplotlib.
        """

        """
        To obtain the directions of the arrows needed, we can take the displacement of each vertex
        to itself. This requires an np.diff() with itself, calculating out[i] = a[i+1] - a[i], and
        then concatenating the last vertex - the first.
        """

        displacement = np.hstack((np.diff(
            self.vertices), (self.vertices[:, 0] - self.vertices[:, -1]).reshape(2, 1)))

        x_values = self.vertices[0, :]
        y_values = self.vertices[1, :]

        x_displacement = displacement[0, :]
        y_displacement = displacement[1, :]

        # if not self.is_filled():
        #     ax = plt.axes()
        #     ax.set_facecolor('dimgray')
        #     plt.fill_between(x_values, y_values, color='white')
        # else:
        #     plt.fill_between(x_values, y_values, color='dimgray')

        plt.quiver(x_values, y_values, x_displacement,
                   y_displacement, scale=1, scale_units='xy', angles='xy', color='black')
        # color = style

        # plt.show()

    def is_filled(self):
        """
        Checks the ordering of the vertices, and returns whether the polygon is filled in or not.
        """

        # Iteratres over the columns of the 2D Matrix to perform the calculation sum((x_2 - x_1) * (y_2 + y_1))
        # If the sum is negative, then the polygon is oriented counter-clockwise, clockwise otherwise.
        num_cols = self.vertices.shape[1]
        running_sum = 0

        for i in range(num_cols - 1):
            x_vals = self.vertices[0, :]
            y_vals = self.vertices[1, :]

            # modulus is for the last element to be compared with the first to close the shape
            running_sum += (x_vals[(i+1) % num_cols] - x_vals[i]) * \
                (y_vals[i] + y_vals[(i+1) % num_cols])

        flag = True if running_sum < 0 else False

        return flag

    def is_self_occcluded(self, idx_vertex, point):
        """
        Given the corner of a polygon, checks whether a given point is self-occluded or not by
        that polygon (i.e., if it is ``inside'' the corner's cone or not). Points on boundary
        (i.e., on one of the sides of the corner) are not considered self-occluded. Note that
        to check self-occlusion, we just need a vertex index  idx_vertex. From this, one can
        obtain the corresponding  vertex, and the  vertex_prev and  vertex_next that precede
        and follow that vertex in the polygon.
        """

        solid = self.is_filled()
        # if idx_vertex == 0, -1 in python refers to the last so it works
        prev_vertex = np.vstack(self.vertices[:, idx_vertex - 1])
        vertex = np.vstack(self.vertices[:, idx_vertex])
        # if idx is the end, we need to loop around to the beginning, defined by the number of columns
        next_vertex = np.vstack(self.vertices[:,
                                              (idx_vertex + 1) % self.vertices.shape[1]])

        # Ensure the vertices are all different
        if (np.array_equal(prev_vertex, vertex) or np.array_equal(next_vertex, vertex)):
            return False

        # Solid Case
        """
            GOAL: If orientation hits prev_vertex first, we are self-occluded
            Compute signed angle between prev and next vertex:
                case 1: signed angle is negative
                    compute signed angle from prev -> point
                    if the sign negative:
                        if angle is >= first_angle then it's occluded
                        else good
                    if the sign is positive:
                        good


                case 2: signed angle is positive
                    compute signed angle from prev -> point
                    if the sign is positive:
                        angle is <= first_angle, then good
                        else occluded
                    if the sign is negative:
                        occluded

        """
        prev_next_angle = angle(vertex, prev_vertex, next_vertex)
        prev_point_angle = angle(vertex, prev_vertex, point)

        flag_point = False

        if solid:
            if (prev_next_angle < 0):
                if (prev_point_angle < 0):
                    flag_point = prev_point_angle >= prev_next_angle
                else:
                    flag_point = False
            else:
                if (prev_point_angle > 0):
                    flag_point = prev_point_angle > prev_next_angle
                else:
                    flag_point = True
        else:
            # Hollow Case
            """
            GOAL: If orientation hits prev_vertex first, we are self-occluded
            Compute the signed angle between prev and the next vertex:
                case 1: signed angle is positive:
                    compute signed angle from prev -> point
                    if the sign is positive:
                        angle <= first_angle, good
                        else, self-occluded
                    if the sign is negative:
                        self-occluded

                case 2: signed angle is negative:
                    compute the signed angle from prev -> point
                    if the sign is negative:
                        angle > first_angle, self-occluded
                        else, good
                    if the sign is positive:
                        good
            """
            if (prev_next_angle > 0):
                if (prev_point_angle > 0):
                    flag_point = prev_point_angle > prev_next_angle
                else:
                    flag_point = True
            else:
                if (prev_point_angle < 0):
                    flag_point = prev_point_angle > prev_next_angle
                else:
                    flag_point = False

        return flag_point

    def is_visible(self, idx_vertex, test_points):
        """
        Checks whether a point p is visible from a vertex v of a polygon. In order to be visible,
        two conditions need to be satisfied: enumerate  point p should not be self-occluded with
        respect to the vertex v (see Polygon.is_self_occluded). The segment p--v should not collide
        with any of the edges of the polygon (see Edge.is_collision).
        """

        vertex = np.vstack(self.vertices[:, idx_vertex])

        flag_points = []
        for point in test_points.T:
            point = np.vstack(point)
            edge_to_check = Edge(np.hstack((vertex, point)))
            if (not self.is_self_occcluded(idx_vertex, point) and no_edge_collisions(self, edge_to_check)):
                flag_points.append(True)
            else:
                flag_points.append(False)

        return flag_points

    def is_collision(self, test_points):
        """
        Checks whether the a point is in collsion with a polygon (that is, inside for a filled in
        polygon, and outside for a hollow polygon). In the context of this homework, this function
        is best implemented using Polygon.is_visible.
        """
        visible_points = np.zeros(test_points.shape[1])
        flag_points = []

        for i in range(self.vertices.shape[1]):
            visible_list = self.is_visible(i, test_points)
            for j in range(test_points.shape[1]):
                if visible_list[j] and visible_points[j] != 1:
                    visible_points[j] = 1

        for val in visible_points:
            if val == 1:
                flag_points.append(False)
            else:
                flag_points.append(True)

        return flag_points


class Edge:
    """ Class for storing edges and checking collisions among them. """

    def __init__(self, vertices):
        """
        Save the input coordinates to the internal attribute vertices.
        """
        self.vertices = vertices

    def is_collision(self, edge):
        """
        Returns  True if the two edges intersect.  Note: if the two edges overlap but are colinear,
        or they overlap only at a single endpoint, they are not considered as intersecting (i.e.,
        in these cases the function returns  False). If one of the two edges has zero length, the
        function should always return the result that edges are non-intersecting.
        """

        # Check to make sure the edge isn't length 0
        if (np.linalg.norm(np.diff(edge.vertices) == 0)):
            return False

        # swap vertices to use diff
        swap_vertices = edge.vertices
        swap_vertices[:, [1, 0]] = swap_vertices[:, [0, 1]]

        A_matrix = np.hstack((np.diff(self.vertices), np.diff(swap_vertices)))

        p1 = np.vstack(self.vertices[:, 0])
        p2 = np.vstack(self.vertices[:, 1])

        p3 = np.vstack(edge.vertices[:, 0])
        p4 = np.vstack(edge.vertices[:, 1])

        # Lines are collinear and can be tested for overlap vs parallelism
        if (np.linalg.det(A_matrix) == 0):
            if (between_segment(p1, p3, p2) or
                    between_segment(p1, p4, p2) or
                    between_segment(p3, p1, p4) or
                    between_segment(p3, p2, p4)):
                return False
            else:
                return True

        b = p3 - p1

        segment_timings = np.linalg.solve(A_matrix, b)
        segment_1_t = abs(segment_timings[0, 0])
        segment_2_u = abs(segment_timings[1, 0])

        tol = 2.22e-16
        # Cases for times:
        t_is_endpoint = (segment_1_t == 0 or segment_1_t == -1*tol or segment_1_t ==
                         tol) or (segment_1_t == 1-tol or segment_1_t == 1 or segment_1_t == 1+tol)
        t_on_segment = segment_1_t >= (-1 * tol) and segment_1_t <= 1 + tol

        u_is_endpoint = (segment_2_u == 0 or segment_2_u == -1*tol or segment_2_u ==
                         tol) or (segment_2_u == 1-tol or segment_2_u == 1 or segment_2_u == 1+tol)
        u_on_segment = segment_2_u >= (-1 * tol) and segment_2_u <= 1 + tol

        # Corner cases
        # 1 Endpoint touching Endpoint
        # 2 Endpoint touching line (T-like shape)
        if (t_is_endpoint and u_is_endpoint):
            return False
        elif (t_on_segment and u_is_endpoint) or (t_is_endpoint and u_on_segment):
            return False
        elif (t_on_segment and u_on_segment):
            return True
        else:
            return False
