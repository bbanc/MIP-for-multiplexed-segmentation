import skimage
import numpy as np

    
def get_compactness(region):
    return 4 * np.pi * region.area / (region.perimeter**2)

def get_rectangularity(region):
    minr, minc, maxr, maxc = region.bbox
    bb_area = (maxr - minr) * (maxc - minc)
    return region.area / bb_area


class ShapeLevelMetrics:
    # Make modular?
    def __init__(self, labels = None, _id = None):
        self.labels = labels
        self.region_features = None
        self._id = _id
        self.did_calc = False
    
    def calc_stats(self):
        if self.labels is not None:
            self.region_features = skimage.measure.regionprops(self.labels)
            
        else: 
            print("Please set labels")
            return

        self.label_IDs = [reg.label for reg in self.region_features]
        self.perimeters = [reg.perimeter for reg in self.region_features]
        self.areas = [reg.area for reg in self.region_features]
        self.solidities = [reg.solidity for reg in self.region_features]
        self.eccentricities = [reg.eccentricity for reg in self.region_features]
        self.major_axes = [reg.axis_major_length for reg in self.region_features]
        self.minor_axes = [reg.axis_minor_length for reg in self.region_features]
        self.compactnesses = [get_compactness(reg) for reg in self.region_features if reg.perimeter > 50]
        self.ars = self.get_ars()
        self.n_cells = self.labels.max()
        self.rectangularities = [get_rectangularity(reg) for reg in self.region_features if reg.area > 50]
        self.density = self.get_density()

    def get_ars(self):
        major_axes = np.array(self.major_axes)
        minor_axes = np.array(self.minor_axes)
    
        # Initialize an array for aspect ratios
        ars = np.zeros_like(major_axes)
    
        # Calculate aspect ratios, handling the case where minor_axes is zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ars = np.where(minor_axes > 0, major_axes / minor_axes, 0)  # or use 0 or None as needed
    
        return ars.tolist() 

    def get_density(self):
        h, w = self.labels.shape
        n_cells = self.labels.max()
        return n_cells / (h*w)
        
    def write_dict(self):
        if not self.did_calc:
            self.calc_stats()
            self.did_calc = True
            
        ret = { "ID": [self._id],
                "Area" : [np.mean(self.areas)],
                "Perimeter" : [np.mean(self.perimeters)],
                "Solidity" : [np.mean(self.solidities)],
                "Eccentricity" : [np.mean(self.eccentricities)],
                "Major Axis" : [np.mean(self.major_axes)],
                "Minor Axis" : [np.mean(self.minor_axes)],
                "Compactness" : [np.mean(self.compactnesses)],
                "Aspect Ratios" : [np.mean(self.ars)],
        		"# Cells" : [self.n_cells],
                "Rectangularity" : [np.mean(self.rectangularities)],
                "Density" : [self.density]
        }
        return ret
        
    def write_dict_stddevs(self):
        if not self.did_calc:
            self.calc_stats()
            self.did_calc = True
            
        ret = { "ID": [self._id],
                "Area" : [np.std(self.areas)],
                "Perimeter" : [np.std(self.perimeters)],
                "Solidity" : [np.std(self.solidities)],
                "Eccentricity" : [np.std(self.eccentricities)],
                "Major Axis" : [np.std(self.major_axes)],
                "Minor Axis" : [np.std(self.minor_axes)],
                "Compactness" : [np.std(self.compactnesses)],
                "Aspect Ratios" : [np.std(self.ars)]

        }
        
        return ret

    def write_dict_max(self):
        if not self.did_calc:
            self.calc_stats()
            self.did_calc = True
            
        ret = { "ID": [self._id],
                "Area" : [np.max(self.areas)],
                "Perimeter" : [np.max(self.perimeters)],
                "Solidity" : [np.max(self.solidities)],
                "Eccentricity" : [np.max(self.eccentricities)],
                "Major Axis" : [np.max(self.major_axes)],
                "Minor Axis" : [np.max(self.minor_axes)],
                "Compactness" : [np.max(self.compactnesses)],
                "Aspect Ratios" : [np.max(self.ars)]

        }
        
        return ret


    def write_dict_p95(self):
        if not self.did_calc:
            self.calc_stats()
            self.did_calc = True
            
        ret = { "ID": [self._id],
                "Area" : [np.percentile(self.areas, 95)],
                "Perimeter" : [np.percentile(self.perimeters, 95)],
                "Solidity" : [np.percentile(self.solidities, 95)],
                "Eccentricity" : [np.percentile(self.eccentricities, 95)],
                "Major Axis" : [np.percentile(self.major_axes, 95)],
                "Minor Axis" : [np.percentile(self.minor_axes, 95)],
                "Compactness" : [np.percentile(self.compactnesses, 95)],
                "Aspect Ratios" : [np.percentile(self.ars, 95)]
        }
        
        return ret

    def write_dict_min(self):
        if not self.did_calc:
            self.calc_stats()
            self.did_calc = True
            
        ret = { "ID": [self._id],
                "Area" : [np.min(self.areas)],
                "Perimeter" : [np.min(self.perimeters)],
                "Solidity" : [np.min(self.solidities)],
                "Eccentricity" : [np.min(self.eccentricities)],
                "Major Axis" : [np.min(self.major_axes)],
                "Minor Axis" : [np.min(self.minor_axes)],
                "Compactness" : [np.min(self.compactnesses)],
                "Aspect Ratios" : [np.min(self.ars)]
        }
        
        return ret



