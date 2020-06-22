#!/usr/bin/python
# -*- coding: utf-8 -*-

# Daniel González Arribas (dangonza@ing.uc3m.es)

import numpy as np

G = 6.67384e-11  # Gravitational constant


norm = np.linalg.norm


def string_to_list(s):
    """Splits the input string by whitespace, returning a list of non-whitespace components


    Parameters
    ----------
    s : string
        String to split

    Returns
    -------
    list
        Non-whitespace string chunks
    """
    return list(filter(lambda x: x, s.strip().split(' ')))


def row_wise_dot(A, B):
    """Row-wise dot product

    Parameters
    ----------
    A : array_like
        2D array
    B : array_like
        2D array with same shape as A

    Returns
    -------
    array_like
        1D array with shape equal to A.shape[0] and B.shape[0]
    """
    return (A * B).sum(axis=1)


def Dyad(a, b):
    """
    Parameters
    ----------
    a : array_like
        3D vector
    b : array_like
        3D vector
    Returns
    -------
    array_like
        Dyadic product of the inputs
    """
    assert a.shape == (3,)
    assert b.shape == (3,)
    return np.tensordot(a, b, axes=0)


class Polyhedron(object):
    """Polyhedron class, equipped with methods for gravitational calculations

    """
    def __init__(self, vertices, face_indexes, density):
        """
        Parameters
        ----------
        vertices : sequence of array_like
            Coordinates of the vertices
        face_indexes : sized of sequences
            A sequence of the form [(a_0, b_0, c_0), (a_1, b_1, c_1), ...] where a_i, b_i, c_i are the indexes of the
            three vertices that compose the polyhedron face
        density : float
            The density of the polyhedron
        """
        self.d = density
        self.points = np.stack(vertices, axis=0)
        self.face_indexes = face_indexes
        self.n_faces = len(face_indexes)
        self.n_edges = 3*(self.n_faces // 2)
        self.faces = np.zeros((self.n_faces, 3, 3), dtype=np.float64)
        self.edges = np.zeros((self.n_edges, 2, 3), dtype=np.float64)
        self.edge_indexes = np.zeros((self.n_edges, 2), dtype=int)
        self.Ff = np.zeros((self.n_faces, 3, 3), dtype=np.float64)
        self.edge_index_to_face_indexes = - np.ones((self.n_edges, 2), dtype=int)
        self.included_edges = []

        edge_idx = 0
        for face_idx, face_points in enumerate(self.face_indexes):
            edge1 = (face_points[0], face_points[1])
            edge2 = (face_points[1], face_points[2])
            edge3 = (face_points[2], face_points[0])
            for edge in (edge1, edge2, edge3):
                s_edge = set(edge)
                try:
                    idx = self.included_edges.index(s_edge)
                    self.edge_index_to_face_indexes[idx, 1] = face_idx
                except ValueError:
                    self.included_edges.append(set(edge))
                    self.edge_index_to_face_indexes[edge_idx, 0] = face_idx
                    self.edge_indexes[edge_idx, :] = edge
                    edge_idx += 1

        for i, point_indexes in enumerate(self.edge_indexes):
            for j, p_idx in enumerate(point_indexes):
                self.edges[i, j, :] = self.points[p_idx, :]

        for i, point_indexes in enumerate(self.face_indexes):
            for j, p_idx in enumerate(point_indexes):
                self.faces[i, j, :] = self.points[p_idx, :]

        # Computing the sign of the first normal
        f = self.faces[0, :, :]
        u = f[1, :] - f[0, :]
        v = f[2, :] - f[1, :]
        w = np.cross(u, v)
        # Assumption: 0 is *inside* the polyhedron
        a = w @ (f[0, :] - np.array([0, 0, 0]))
        self.normals_sign = np.sign(a)
        if not self.normals_sign or np.isnan(self.normals_sign):
            # In the rare case that f[0, :] == Vector([0,0,0])
            eps = 2 * np.finfo(np.float64).eps
            a = w @ (f[0, :] - np.array([eps, eps, eps]))
            self.normals_sign = a/abs(a)

        # Compute normals
        self.face_normals = np.zeros((self.n_faces, 3), dtype=np.float64)
        for i, face in enumerate(self.faces):
            u = face[1, :] - face[0, :]
            v = face[2, :] - face[1, :]
            w = np.cross(u, v)
            w_hat = w/norm(w)
            self.face_normals[i, :] = w_hat * self.normals_sign

        # Compute edge dyads
        self.Ee = np.zeros((self.n_edges, 3, 3), dtype=np.float64)
        for edge_idx, edge_points in enumerate(self.edge_indexes):
            edge = self.edges[edge_idx, :, :]
            edge_vector = edge[1, :] - edge[0, :]
            fA, fB = self.edge_index_to_face_indexes[edge_idx, :]
            nA = self.face_normals[fA, :]
            nB = self.face_normals[fB, :]
            unit_edge = edge_vector/norm(edge_vector)
            face_A = self.face_indexes[fA]
            nA12, nB21 = None, None
            for i in range(3):
                if face_A[i] not in edge_points:
                    if face_A[(1 + i) % 3] == edge_points[0]:
                        nA12 = np.cross(nA, unit_edge) * (-1)
                        nB21 = np.cross(nB, unit_edge)
                    else:
                        nA12 = np.cross(nA, unit_edge)
                        nB21 = np.cross(nB, unit_edge) * (-1)
            self.Ee[edge_idx, :, :] = (Dyad(nA, nA12) + Dyad(nB, nB21))

        # Compute face dyads
        for face_idx, face_normal in enumerate(self.face_normals):
            self.Ff[face_idx, :, :] = Dyad(face_normal, face_normal)

    @classmethod
    def init_from_obj_file(cls, f, scale=1, density=1):
        """
        Initializes the  from an .obj file

        Parameters:
        -----------
        f : file_like
            input obj file. For format, see test samples
        scale : float
            scale coordinates by this amount when reading from file
        density : float
            density of the polyhedron
        """
        lines = [line.strip() for line in f.readlines()]
        vertices = []
        indexes = []
        for line in lines:
            if line.startswith("v"):  # vertex
                nums = list(map(float, string_to_list(line[2:])))
                vertices.append(scale * np.array(nums[:3]))
                # x.append(nums[0] * scale)
                # y.append(nums[1] * scale)
                # z.append(nums[2] * scale)
            elif line.startswith("f"):  # face
                nums = list(map(lambda a: int(a) - 1, string_to_list(line[2:])))
                indexes.append(nums)
        return cls(vertices, indexes, density)

    def _Le(self, p):
        """Potential of the edges"""
        a = norm(p - self.edges[:, 0, :], axis=1)
        b = norm(p - self.edges[:, 1, :], axis=1)
        le = norm(self.edges[:, 0, :] - self.edges[:, 1, :], axis=1)
        q = (a + b + le) / (a + b - le)  # TODO: set the quotient to 0 whenever (a + b - le) == 0
        return np.log(q)

    def _re(self, p):
        """Position vectors of points in the edges from the field point reference """
        return self.edges[:, 0, :] - p  # 0 is arbitrary - the other end also works

    def _rf(self, p):
        """Position vectors of points in the faces from the field point reference """
        return self.faces[:, 0, :] - p  # 0 is arbitrary - the other vertices also work

    def _wf(self, p):
        """Signed solid angle subtended by the polyhedron faces when viewed from point p"""
        r = self.faces - p
        n = norm(r, axis=2)
        num = row_wise_dot(r[:, 0, :], np.cross(r[:, 1, :], r[:, 2, :]))
        den = n[:, 1] * n[:, 2] * n[:, 0]
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            den += row_wise_dot(r[:, i, :], r[:, j, :]) * n[:, k]
        return 2*np.arctan2(num, den)

    # def delta_z(self, p):
    #     return row_wise_dot(self.face_normals, (self.faces[:, 0, :] - p))

    def U(self, p):
        """Computes the gravitational potential at the given point

        Parameters
        ----------
        p : array_like
            field point at which the potential is computed
        Returns
        -------
        float
            Potential at p
        """
        re = self._re(p)
        Le = self._Le(p)
        wf = self._wf(p)
        rf = self._rf(p)
        A = row_wise_dot(re, np.einsum('...ij,...j', self.Ee, re)) @ Le
        B = row_wise_dot(rf, np.einsum('...ij,...j', self.Ff, rf)) @ wf
        return (B - A) * G * self.d * 0.5

    def g(self, p):
        """Computes the gravitational attraction at the given point

        Parameters
        ----------
        p : array_like
            field point at which the gravity is computed

        Returns
        -------
        array_like
            Attraction vector (with shape (3,)) at the given point
        """
        re = self._re(p)
        Le = self._Le(p)
        wf = self._wf(p)
        rf = self._rf(p)
        A = Le @ np.einsum('...ij,...j', self.Ee, re)
        B = wf @ np.einsum('...ij,...j', self.Ff, rf)
        return (B - A) * G * self.d

    def gravity_gradients(self, p):
        """Computes the gradients of the gravitational attraction at the given point

        Parameters
        ----------
        p : array_like
            field point at which the gravity is computed

        Returns
        -------
        array_like
            Gravity gradients matrix (with shape (3, 3)) at the given point
        """
        Le = self._Le(p)
        wf = self._wf(p)
        EeLe = np.einsum("i...,i", self.Ee, Le)
        FfWf = np.einsum("i...,i", self.Ff, wf)
        return G * self.d * (EeLe - FfWf)

    def laplacian(self, p):
        """Computes the Laplacian of the potential at the given point

        Parameters
        ----------
        p : array_like
            field point at which the Laplacian is computed

        Returns
        -------
        float
            Laplacian at p
        """
        wf = self._wf(p)
        return - G * self.d * wf.sum()





