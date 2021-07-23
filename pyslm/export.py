 
class MTTExportFeature(Feature):
    
            # Combines the raw features into the final TPMS field
    # The lattice grids always override the solid grids
    def __init__(self, hatchFeature, name = 'MTTExporter'):
        
        Feature.__init__(self,name)
        
        
        self._layerThickness  = 10
        self._filename        = ''
        self._models          = []
        
        self._hatchFeature    = hatchFeature
        self._setAttributes([self._hatchFeature])
        
        
        self._value = []
        
        print('Constructed Export to Renishaw')
       
    @staticmethod
    def exportMTT(header, models, layers):
        # The static method is called at the end of processing the collection of layers and exports to the filename
        
        import renishawExport as MTT
        MTT.exportMTT(header, models, layers)               
        
    @staticmethod
    def version():
        return (1,0) # Major, Minor version

    
    @property
    def filename(self):
        return self._filename
             
    def value(self, update = False):

        if not self.requiresRecompute():
            return None
        
        if self.isDirty() or update:
            self.update()   
        
        return self._value
    
    
    def update(self):
        
        import renishawExport as MTT
        
        # Process the models
        
        # Process the layers        
        layerChunk = self._hatchFeature.value()
        
        slmLayers = []
        
        for sliceLayer in layerChunk:
            
            layer         = MTT.Layer()
            layer.layerId = sliceLayer.id
            layer.z       = sliceLayer.z
            
            geoms = []
            
            for contour in sliceLayer.contours:
                layerGeom = MTT.LayerGeometry()
                layerGeom.type = 'contour'
                layerGeom.coords = contour.coords
                layerGeom.bid = 0
                layerGeom.mid = 0
                
                geoms.append(layerGeom)
                

            for hatches in sliceLayer.hatches:
                layerGeom = MTT.LayerGeometry()
                layerGeom.type = 'hatch'
                layerGeom.coords = hatches.coords
                layerGeom.bid = 0
                layerGeom.mid = 0
                
                geoms.append(layerGeom)
                                
            
            for points in sliceLayer.points:
                layerGeom = MTT.LayerGeometry()
                layerGeom.type = 'point'
                layerGeom.coords = points.coords
                layerGeom.bid = 0
                layerGeom.mid = 0
                
                geoms.append(layerGeom)
                
            # Add the layer geometries to the layer            
            layer.geometry = geoms
            
            slmLayers.append(layer)
            
        self._value = slmLayers
