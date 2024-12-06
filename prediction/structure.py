from Bio.PDB import PDBParser, DSSP
import numpy as np
from scipy.spatial import ConvexHull


SS_mapper = {
    "H": "Alpha helix (4-12)",
    "B": "Isolated beta-bridge residue",
    "E": "Strand",
    "G": "3-10 helix",
    "P": "Pi helix",
    "T": "Turn",
    "S": "Bend",
    "-": "None",
    "I": "Irregular secondary structure"
}

SS = {
    "H": [1, 0, 0, 0, 0, 0, 0, 0, 0],
    "B": [0, 1, 0, 0, 0, 0, 0, 0, 0],
    "E": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    "G": [0, 0, 0, 1, 0, 0, 0, 0, 0],
    "P": [0, 0, 0, 0, 1, 0, 0, 0, 0],
    "T": [0, 0, 0, 0, 0, 1, 0, 0, 0],
    "S": [0, 0, 0, 0, 0, 0, 1, 0, 0],
    "-": [0, 0, 0, 0, 0, 0, 0, 1, 0],
    "I": [0, 0, 0, 0, 0, 0, 0, 0, 1]
}


def RSAMapper(RSA):
    a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if RSA <= 0.1:
        a[0] = 1
    elif RSA <= 0.2:
        a[1] = 1
    elif RSA <= 0.3:
        a[2] = 1
    elif RSA <= 0.4:
        a[3] = 1
    elif RSA <= 0.5:
        a[4] = 1
    elif RSA <= 0.6:
        a[5] = 1
    elif RSA <= 0.7:
        a[6] = 1
    elif RSA <= 0.8:
        a[7] = 1
    elif RSA <= 0.9:
        a[8] = 1
    else:
        a[9] = 1
    return a


def get_center_atom(res):
    if 'CA' in res:
        return res['CA']
    elif 'C' in res:
        return res['C']
    elif 'N' in res:
        return res['N']
    elif 'O' in res:
        return res['O']
    elif 'CB' in res:
        return res['CB']
    else:
        return None

def calculate_SS_RSA_torsion(model): #21
    result = []
    for chain in model:
        for residue in chain:
            key = (chain.id,residue.id)
            # print("SS", SS[dssp[key][2]])
            # print("RSA (Relative ASA)", RSAMapper(dssp[key][3]))
            # print("Phi", dssp[key][4])
            # print("Psi", dssp[key][5])
            # print("classical coordination number", exp_fs[key])

            try:
                result.append(SS[dssp[key][2]] + RSAMapper(dssp[key][3]) + [dssp[key][4]] + [dssp[key][5]])
            except:
                result.append(SS['-'] + RSAMapper(0) + [180] + [180])
    return np.array(result)

def calculate_neighbour_count(model, cutoff): #1
    contact_counts = []
    for chain_1 in model:
        for res_1 in chain_1:
            count = 0

            for chain_2 in model:
                for res_2 in chain_2:

                    distance = abs(get_center_atom(res_1) - get_center_atom(res_2))

                    if distance <= cutoff:
                        count += 1

            contact_counts.append(count)

    return np.array(contact_counts).reshape(-1, 1)

def calculate_virtual_surface_area(model): #1
    surface_areas = []
    for chain in model:
        for res in chain:
            atom_coords = []
            for atom in res:
                atom_coords.append(atom.get_coord())

            if len(atom_coords) >= 4:
                hull = ConvexHull(atom_coords)

                # The surface area of the convex hull
                surface_area = hull.area

                # Append the surface area to the list
                surface_areas.append(surface_area)
            else:
                surface_areas.append(0)
    return np.array(surface_areas).reshape(-1, 1)

def calculate_relative_positioning(model): #2
    relative_positions = []

    for chain in model:
        coords = []
        for res in chain:
            coords.append(get_center_atom(res).get_coord())
        # print(np.array(coords).shape)
        centroid = np.mean(np.array(coords), axis=0)
        # print(centroid)
        for i, res in enumerate(chain):
            # Relative sequence position
            sequential_pos = 1 / (i + 1)

            distance = np.linalg.norm(get_center_atom(res).get_coord() - centroid)
            
            if distance != 0:
                spatial_pos = 1 / distance
            else:
                spatial_pos = 0
            relative_positions.append([sequential_pos, spatial_pos])

    return np.array(relative_positions)

def calculate_residue_orientation(model): #9
    orientations = []

    keys = []

    for chain in model:
        for res in chain:
            keys.append(res.get_full_id()[3])

    for chain in model:
        for i in range(0, len(chain)):
            res = chain[keys[i]]
            if i != 0:
                res_prev = chain[keys[i - 1]]
            else:
                res_prev = res
            if i != len(chain) - 1:
                res_next = chain[keys[i + 1]]
            else:
                res_next = res

            # Vectors
            
            ca = get_center_atom(res).get_coord()
            
            ca_prev = get_center_atom(res_prev).get_coord()
            
            ca_next = get_center_atom(res_next).get_coord()
            
            if 'CB' in res:
                cb = res['CB'].get_coord()
            else:
                cb = ca

            vec_prev = ca_prev - ca # i -> i - 1
            vec_next = ca_next - ca # i -> i + 1
            vec = cb - ca # ai -> bi

            if np.linalg.norm(vec_prev) != 0:
                unit_vec_prev = vec_prev / np.linalg.norm(vec_prev)
            else:
                unit_vec_prev = vec_prev
            if np.linalg.norm(vec_next) != 0:
                unit_vec_next = vec_next / np.linalg.norm(vec_next)
            else:
                unit_vec_next = vec_next
            if np.linalg.norm(vec) != 0:
                unit_vec = vec / np.linalg.norm(vec)
            else:
                unit_vec = vec

            # print(unit_vec_prev.tolist())
            # print(unit_vec_next)
            # print(unit_vec)
            orientations.append(unit_vec_prev.tolist() + unit_vec_next.tolist() + unit_vec.tolist())

    return np.array(orientations)

def calculate_cosine_angle(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def compute_local_geometry(model): #2
    local_geometry = []

    keys = []

    for chain in model:
        for res in chain:
            keys.append(res.get_full_id()[3])

    for chain in model:
        for i in range(0, len(chain)):
            res = chain[keys[i]]
            if i != 0:
                res_prev = chain[keys[i - 1]]
            else:
                res_prev = res
            if i != len(chain) - 1:
                res_next = chain[keys[i + 1]]
            else:
                res_next = res

            try:
                co_prev = res_prev['O'].get_coord() - res_prev['C'].get_coord()
                co = res['O'].get_coord() - res['C'].get_coord()

                cosine_angle = calculate_cosine_angle(co_prev, co)
            except:
                cosine_angle = 0

            curr = get_center_atom(res).get_coord()
            
            prev = get_center_atom(res_prev).get_coord()
            
            next = get_center_atom(res_next).get_coord()

            vec_bond1 = curr - prev # i-1 -> i
            vec_bond2 = next - curr # i -> i+1
            bond_angle = calculate_cosine_angle(vec_bond1, vec_bond2)

            local_geometry.append([cosine_angle, bond_angle])
    return np.array(local_geometry)

def get_coords(model):
    coords = []
    for chain in model:
        for res in chain:
            coords.append(get_center_atom(res).get_coord())
    return np.array(coords)