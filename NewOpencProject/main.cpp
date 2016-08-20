#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
//#include "descriptor.h"
#include "ReidDescriptor.h"
#include "histdescriptor.h"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[])
{
    // /Users/dougalasmichael/Documents/ReidDatasets/prid_2011/single_shot
    string inputFolder  = "/Users/JohnsonJohnson/Documents/ReidDatasets/VIPeR/";
  //  string inputFolder  = "/Users/dougalasmichael/Documents/ReidDatasets/VIPeR/";
        
        string defaultProbesFilename  = "probes.txt";
        string defaultGalleryFilename = "gallery.txt";
        string defaultOutputFilename  = "table.csv";
    
   if (argc > 1)
       inputFolder = argv[1];
    
   for (int i = 2; i < argc; i++)
   {
       if (i + 1 != argc)
       {
           if (string(argv[i]) == "-p")
           {
               defaultProbesFilename = argv[i + 1];
           }
           if (string(argv[i]) == "-g")
           {
               defaultGalleryFilename = argv[i + 1];
           }
           if (string(argv[i]) == "-o")
           {
               defaultOutputFilename = argv[i + 1];
           }
           
       }
   }
    
    printf("Dataset: %s\n", inputFolder.c_str());
    printf("probes file: %s\n", defaultProbesFilename.c_str());
    printf("gallery file: %s\n", defaultGalleryFilename.c_str());
    printf("output file: %s\n", defaultOutputFilename.c_str());
    
   
   string output = defaultOutputFilename;
    
   vector<string> probes;
   vector<string> gallery;
    
   TextInput::readLines(inputFolder + defaultProbesFilename, probes);
   TextInput::readLines(inputFolder + defaultGalleryFilename, gallery);
    
   //Implement your own descriptor and change the variable desc
    Ptr<Descriptor> desc = new ReID();
  // Ptr<Descriptor> desc = new DescriptorY();

    
   SimilarityTable table(desc, inputFolder, probes, gallery);
   table.createTable();
   table.outputFile(output);
    
  
   return 0;
    
}