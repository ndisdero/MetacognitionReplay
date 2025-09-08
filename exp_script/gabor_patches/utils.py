import numpy as np

def generate_orientation_patches(num_patches, mean_orientation, sd):
    orientation_patches = np.random.normal(loc=mean_orientation, scale=sd, size=num_patches)
    return orientation_patches

def generate_position_patches(num_patches, radius, distance_min):
    # Generate random positions for the patches within specified ranges
    # BUT it must ensure that the patches do not overlap
    # Furthermore, it should ensure that the generation of positions is not
    # in a square grid, but a circle. 
    rng = np.random.default_rng(124)
    positions = []
    for _ in range(num_patches):
        while True:
            x = rng.uniform(-radius, radius)
            y = rng.uniform(-radius, radius)
            
            # Distance from the center should be less than the radius from 0
            distance_from_center = np.sqrt((x - 0)**2 +(y - 0)**2)
            if distance_from_center > radius:
                continue
            # And the position should not overlap with existing positions
            position = (x, y)
            if not any(np.linalg.norm(np.array(position) - np.array(p)) < distance_min for p in positions):
                positions.append(position)
                break
    return positions

if __name__ == "__main__":
    # Example usage
    num_patches = 15
    mean_orientation = 90
    sd = 10
    orientations = generate_orientation_patches(num_patches, mean_orientation, sd)
    print("Generated Orientations:", orientations)

    positions = generate_position_patches(num_patches, radius=200)
    print("Generated Positions:", positions)