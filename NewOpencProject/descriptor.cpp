#include "descriptor.h"

using namespace std;

double Descriptor::bhattacharyya(cv::Mat k, cv::Mat q)
{
    cv::normalize(k, k, 1, 0, cv::NORM_L1);
    cv::normalize(q, q, 1, 0, cv::NORM_L1);
    
    cv::Mat temp = k.mul(q);
    sqrt(temp, temp);
    
    return (double)sqrt(1 - cv::sum(temp)[0]);
    // sqrt(1-sum(sqrt(k.*q)))
}


SimilarityTable::SimilarityTable(const Ptr<Descriptor> &desc,
                const string &folder,
                const vector<string> &probes,
                const vector<string> &gallery) :
_folder(folder), _probes(probes), _gallery(gallery), _descriptor(desc)
{
    size_t m = gallery.size(); //number of images in gallery
    size_t n = probes.size();   //number of probes
    
    _similarityTable = Mat(m, n, CV_64FC1, Scalar(std::numeric_limits<float>::max()));
    
}

void SimilarityTable::createTable()
{
    size_t n = _probes.size();
    size_t m = _gallery.size();
    std::cout.precision(2);
    for (size_t c = 0; c < n; c++)
    {
        Mat probeItem = cv::imread(_folder + _probes[c]);
        
        string probe_file = _probes[c].c_str();
        cout << "Probe:" << setw(6) << std::setfill(' ')  << c << " "
        << "Total:" << setw(4) << std::setfill(' ')  << (c+1.)/float(n) * 100 << " %"
        << setw(8) << std::setfill(' ')  << endl;
        
        
        for (size_t r = 0; r < m; r++)
        {
            Mat galleryItem = cv::imread(_folder + _gallery[r]);
            
            _similarityTable.at<double>(r,c) =
            saturate_cast<double>(_descriptor->distance(probeItem, galleryItem));
            
        }
    }
    
    
}

void SimilarityTable::outputFile(const string &filename)
{
    ofstream oFile;
    oFile.open(filename);
    oFile << format(_similarityTable, "csv");
    oFile.close();
}


void TextInput::readLines(const string& filename, vector<string> &lines)
{
    std::ifstream infile(filename);
    
    std::string line;
    while (std::getline(infile, line))
    {
     //   cout<<line<<endl;
        lines.push_back(line);
    }
    infile.close();
}
