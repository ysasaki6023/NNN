#include <iostream>
#undef __DEBUG__
#include <Python.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include "algorithm"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

using namespace std;

////////////////////////////////////////////
////////////////////////////////////////////
class NNN
{
    public:
    //
    NNN();
    ~NNN();
    void test();
    void OpenOutputfile(char* filename, bool CreateNewfile);
    void CloseOutputfile();
    void DumpToFile(unsigned int totalIter, unsigned int sampleIndex, unsigned int iStep, char* comment);
    void ApplyDeltaW();
    //template <class U> void DumpToFile1D(const char* title, U ptr[], unsigned int range_from, unsigned int range_to);
    //template <class U> void DumpToFile2D(const char* title, U ptr[], unsigned int range_from, unsigned int range_to);
    template <class U> void DumpToFile1D(const char* title, U ptr[], vector<unsigned int> *DumpIndexVec, bool DumpTitle);
    template <class U> void DumpToFile2D(const char* title, U ptr[], vector< pair<unsigned int, unsigned int> > *DumpIndexVec, bool DumpTitle);
    void SetRecordTarget1D(char* targetName, unsigned int tgtIdx);
    void SetRecordTarget2D(char* targetName, unsigned int from, unsigned int to);
    void AllocNeurons(unsigned int m_N_nodes_input, unsigned int m_N_nodes_output, unsigned int m_N_nodes_hidden);
    int* LearnOnce(unsigned int SampleNumber  , unsigned int nSteps, char* comment, bool DoInitialize=true, bool DoLearning = true);
    //void SetSamples(vector< vector<double>* >* m_Samples);
    void SetSamples(vector< vector<int>* >* m_Samples_input,vector< vector<int>* >* m_Samples_output);
    void PrintR(bool b_input=true, bool b_output=true, bool b_hidden=true);
    void PrintX(bool b_input=true, bool b_output=true, bool b_hidden=true);
    void PrintS(bool b_input=true, bool b_output=true, bool b_hidden=true);
    void PrintW();
    void SetXsigma(double m_sigma);
    void SetKupdate(double m_Kupdate);
    void SetXupdateLambda(double m_XupdateLambda);
    double   K_update;
    double   K_rand_Xsigma;
    double   K_rand_Wsigma;
    double   K_XupdateLambda;
    void SetConnection(unsigned int from, unsigned int to);
    void Initialize();
    private:
    //
    vector< vector< int > * > *p_Samples_input, *p_Samples_output;
    unsigned int N_iteration;
    unsigned int N_nodes_input, N_nodes_output, N_nodes_inout, N_nodes_hidden, N_nodes_total;
    double  *W_t;//, W_tm1;
    double  *deltaW_int;;
    int     *R_t, *R_tm1; // Digitized results
    double  *X_t, *X_tm1; // Analog results
    double  *S1;
    double  *S2;
    vector< vector<int> > Connection_from_to;
    vector< vector<int> > Connection_to_from;
    //
    inline int ConvThreshold(double val);
    inline double rand_uniform();
    inline double rand_normal(double mu, double sigma);
    //
    ofstream outfile;
    bool DoRequireHeader;
    vector<unsigned int> *p_DumpList_X, *p_DumpList_R;
    vector< pair<unsigned int,unsigned int> > *p_DumpList_W;
    
};
////////////////////////////////////////////
void NNN::OpenOutputfile(char* filename, bool CreateNewfile)
{
    if(CreateNewfile)
    {
        outfile.open(filename,ios::out);
        DoRequireHeader = true;
    }
    else
    {
        outfile.open(filename,ios::out|ios::app);
        DoRequireHeader = false;
    }

    return;
}
////////////////////////////////////////////
void NNN::CloseOutputfile()
{
    outfile.close();
    return;
}
////////////////////////////////////////////
void NNN::SetRecordTarget1D(char* targetName, unsigned int tgtIdx)
{
    string tgtName = targetName;
    if(tgtName=="X")
    {
        p_DumpList_X->push_back( tgtIdx );
    }
    else if(tgtName=="R")
    {
        p_DumpList_R->push_back( tgtIdx );
    }
    else
    {
        cout<<"Wrong targetName"<<endl;
    }
}
////////////////////////////////////////////
void NNN::SetRecordTarget2D(char* targetName, unsigned int from, unsigned int to)
{
    string tgtName = targetName;
    if(tgtName=="W")
    {
        p_DumpList_W->push_back( make_pair(from,to) );
    }
    else
    {
        cout<<"Wrong targetName"<<endl;
    }
}
////////////////////////////////////////////
void NNN::DumpToFile(unsigned int totalIter, unsigned int sampleIndex, unsigned int iStep, char* comment)
{
    string s_comment = comment;
    if(DoRequireHeader)
    {
        outfile<<"totalIter,sampleIndex,iStep,comment,";
        DumpToFile1D<int>   ("R",R_t,p_DumpList_R,true);
        DumpToFile1D<double>("X",X_t,p_DumpList_X,true);
        DumpToFile2D<double>("W",W_t,p_DumpList_W,true);
        outfile<<"endflag"<<endl;
        DoRequireHeader = false;
    }
        outfile<<totalIter<<","<<sampleIndex<<","<<iStep<<","<<s_comment<<",";
        DumpToFile1D<int>   ("R",R_t,p_DumpList_R,false);
        DumpToFile1D<double>("X",X_t,p_DumpList_X,false);
        DumpToFile2D<double>("W",W_t,p_DumpList_W,false);
        outfile<<"0"<<endl;
    return;
}
////////////////////////////////////////////
template <class U> void NNN::DumpToFile1D(const char* title, U ptr[], vector<unsigned int> *DumpIndexVec, bool DumpTitle)
{
    if(not DumpTitle)
    {
        for(unsigned int i=0; i<DumpIndexVec->size(); ++i)
        {
            outfile<<ptr[(*DumpIndexVec)[i]]<<",";
        }
    }
    else
    {
        for(unsigned int i=0; i<DumpIndexVec->size(); ++i)
        {
            outfile<<title<<"("<<(*DumpIndexVec)[i]<<"),";
        }
    }
    return;
}
////////////////////////////////////////////
template <class U> void NNN::DumpToFile2D(const char* title, U ptr[], vector< pair<unsigned int, unsigned int> > *DumpIndexVec, bool DumpTitle)
{
    if(not DumpTitle)
    {
        for(unsigned int i=0; i<DumpIndexVec->size();++i)
        {
            pair<unsigned int, unsigned int> pp;
            pp = (*DumpIndexVec)[i];
            outfile<<ptr[pp.first*N_nodes_total+pp.second]<<",";
        }
    }
    else
    {
        for(unsigned int i=0; i<DumpIndexVec->size();++i)
        {
            pair<unsigned int, unsigned int> pp;
            pp = (*DumpIndexVec)[i];
            outfile<<title<<"("<<pp.first<<"-"<<pp.second<<"),";
        }
    }
    return;
}
////////////////////////////////////////////
void NNN::SetXsigma(double m_sigma)
{
    K_rand_Xsigma = m_sigma;
    return;
}
////////////////////////////////////////////
void NNN::SetKupdate(double m_Kupdate)
{
    K_update = m_Kupdate;
    return;
}
////////////////////////////////////////////
void NNN::SetXupdateLambda(double m_XupdateLambda)
{
    K_XupdateLambda = m_XupdateLambda;
    return;
}
////////////////////////////////////////////
void NNN::test()
{
    cout<<"test"<<endl;
}
////////////////////////////////////////////
void NNN::SetSamples(vector< vector<int>* > *m_Samples_input,vector< vector<int>* > *m_Samples_output)
{
    p_Samples_input  = m_Samples_input;
    p_Samples_output = m_Samples_output;
    return;
}
////////////////////////////////////////////
void NNN::AllocNeurons(unsigned int m_N_nodes_input, unsigned int m_N_nodes_output, unsigned int m_N_nodes_hidden)
{
    // Set
    N_nodes_input  = m_N_nodes_input;
    N_nodes_output = m_N_nodes_output;
    N_nodes_hidden = m_N_nodes_hidden;
    N_nodes_inout  = m_N_nodes_input + m_N_nodes_output;
    N_nodes_total  = m_N_nodes_input + m_N_nodes_output+ m_N_nodes_hidden;
    //cout<<"total="<<N_nodes_total<<endl;
    K_update       = 0.01;
    K_rand_Xsigma  = 0.01;
    K_rand_Wsigma  = 0.01;
    N_iteration    = 0;
    K_XupdateLambda= 1.0;

    // Alock spaces
    W_t       = new double[N_nodes_total*N_nodes_total];
    deltaW_int= new double[N_nodes_total*N_nodes_total];
    R_t       = new int   [N_nodes_total];
    R_tm1     = new int   [N_nodes_total];
    X_t       = new double[N_nodes_total];
    X_tm1     = new double[N_nodes_total];
    S1        = new double[N_nodes_total];
    S2        = new double[N_nodes_total*N_nodes_total];
    Connection_from_to.resize(N_nodes_total);
    Connection_to_from.resize(N_nodes_total);

    // Initialize
    for(unsigned int i=0; i<N_nodes_total; ++i)
    {
        for(unsigned int j=0; j<N_nodes_total; ++j)
        {
            W_t[i*N_nodes_total+j] = 0.;
            deltaW_int[i*N_nodes_total+j] = 0.;
            if(i==j)
            {
                S2 [i*N_nodes_total+j] = 1.;
            }
            else
            {
                S2 [i*N_nodes_total+j] = 0.;
            }
        }
        R_t  [i]   = 0.;
        R_tm1[i]   = 0.;
        X_t  [i]   = 0.;
        X_tm1[i]   = 0.;
        S1   [i]   = 0.;
    }
    return;
}
////////////////////////////////////////////
void NNN::SetConnection(unsigned int from, unsigned int to)
{
    if(from==to)
    {
        cout<<"Self-connection is prohibited"<<endl;
        return;
    }

    if(find(Connection_from_to[from].begin(),Connection_from_to[from].end(),to)==Connection_from_to[from].end())
    {
        Connection_from_to[from].push_back(to);
        Connection_to_from[to]  .push_back(from);
    }
    else
    {
        cout<<"Connection exists already"<<endl;
    }
}
////////////////////////////////////////////
int NNN::ConvThreshold(double val)
{
    if(val>=0.) return +1;
    else        return -1;
}
////////////////////////////////////////////
void NNN::PrintS(bool b_input, bool b_output, bool b_hidden)
{
    cout<<endl;
    cout<<"S values (N_Iteration = "<<N_iteration<<" )"<<endl;
    if(b_input)
    {
        for(unsigned int i=0; i<N_nodes_input; ++i)
        {
            cout<<"S("<<i<<", in     ) = "<<(double)S1[i]/N_iteration<<endl;
        }
    }
    if(b_output)
    {
        for(unsigned int i=N_nodes_input; i<N_nodes_inout; ++i)
        {
            cout<<"S("<<i<<", out    ) = "<<(double)S1[i]/N_iteration<<endl;
        }
    }
    if(b_hidden)
    {
        for(unsigned int i=N_nodes_inout; i<N_nodes_total; ++i)
        {
            cout<<"S("<<i<<", hidden ) = "<<(double)S1[i]/N_iteration<<endl;
        }
    }
    cout<<endl;
}
////////////////////////////////////////////
void NNN::PrintR(bool b_input, bool b_output, bool b_hidden)
{
    cout<<endl;
    cout<<"R values (N_Iteration = "<<N_iteration<<" )"<<endl;
    if(b_input)
    {
        for(unsigned int i=0; i<N_nodes_input; ++i)
        {
            cout<<"R("<<i<<", in     ) = "<<R_t[i]<<endl;
        }
    }
    if(b_output)
    {
        for(unsigned int i=N_nodes_input; i<N_nodes_inout; ++i)
        {
            cout<<"R("<<i<<", out    ) = "<<R_t[i]<<endl;
        }
    }
    if(b_hidden)
    {
        for(unsigned int i=N_nodes_inout; i<N_nodes_total; ++i)
        {
            cout<<"R("<<i<<", hidden ) = "<<R_t[i]<<endl;
        }
    }
    cout<<endl;
}
////////////////////////////////////////////
void NNN::PrintX(bool b_input, bool b_output, bool b_hidden)
{
    cout<<endl;
    cout<<"X values (N_Iteration = "<<N_iteration<<" )"<<endl;
    if(b_input)
    {
        for(unsigned int i=0; i<N_nodes_input; ++i)
        {
            cout<<"X("<<i<<", in     ) = "<<X_t[i]<<endl;
        }
    }
    if(b_output)
    {
        for(unsigned int i=N_nodes_input; i<N_nodes_inout; ++i)
        {
            cout<<"X("<<i<<", out    ) = "<<X_t[i]<<endl;
        }
    }
    if(b_hidden)
    {
        for(unsigned int i=N_nodes_inout; i<N_nodes_total; ++i)
        {
            cout<<"X("<<i<<", hidden ) = "<<X_t[i]<<endl;
        }
    }
    cout<<endl;
}
////////////////////////////////////////////
void NNN::PrintW()
{
    cout<<endl;
    cout<<"W values (N_Iteration = "<<N_iteration<<" )"<<endl;
    for(unsigned int i=0; i<N_nodes_total; ++i)
    {
        for(unsigned int j=0; j<N_nodes_total; ++j)
        {
            cout<<"W("<<i<<" -> "<<j<<") = "<<W_t[i*N_nodes_total+j]<<endl;
        }

    }
    cout<<endl;
}
////////////////////////////////////////////
double NNN::rand_uniform()
{
    double ret = ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);
    return ret;
}
////////////////////////////////////////////
double NNN::rand_normal(double mu, double sigma)
{
    double  z = sqrt(-2.0 * log(rand_uniform())) * sin(2.0 * M_PI * rand_uniform());
    return mu + sigma * z;
}
////////////////////////////////////////////
void NNN::Initialize()
{
    // Set random
    srand(101);

    // Randomize connections
    for(unsigned int i=0; i<N_nodes_total; ++i)
    {
        for(vector<int>::iterator jIter=Connection_to_from[i].begin();jIter!=Connection_to_from[i].end(); ++jIter)
        {
            W_t[i*N_nodes_total+(*jIter)] = rand_normal(0., K_rand_Wsigma);
            deltaW_int[i*N_nodes_total+(*jIter)] = 0.;
        }
    }

}
////////////////////////////////////////////
int* NNN::LearnOnce(unsigned int SampleNumber  , unsigned int nSteps       ,char* comment, bool DoInitialize,      bool DoLearning)
{
    int *R_input, *R_output;
    R_input = &(*((*p_Samples_input)[SampleNumber]))[0];
    if(p_Samples_output->size()>0) R_output = &(*((*p_Samples_output)[SampleNumber]))[0];
    else R_output = NULL;

    vector< int >::iterator         jIter,kIter;

    if(DoInitialize)
    {
        for(unsigned int j=0; j<N_nodes_total; ++j)
        {
            X_tm1[j] = 2.*rand_uniform()-1.;
            R_tm1[j] = ConvThreshold(X_tm1[j]);
            X_t[j]   = 2.*rand_uniform()-1.;
            R_t[j]   = ConvThreshold(X_t  [j]);
        }
    }

    for(unsigned int iStep = 0; iStep<nSteps; ++iStep)
    {
        N_iteration ++;
        // Swap and Cleanup
        for(unsigned int j=0; j<N_nodes_total; ++j)
        {
            // Swap
            X_tm1[j] = X_t[j];
            R_tm1[j] = R_t[j];
            // Do not clean up the current numbers. Will be reffered later!!
        }
        // Update X and R
        // 非同期更新
        // Update output + hidden layer
        for(unsigned int i=N_nodes_input; i<N_nodes_total; ++i)
        {
            // Accumulate
            X_t[i] = 0.;
            for(jIter=Connection_to_from[i].begin();jIter!=Connection_to_from[i].end(); ++jIter)
            {
                if((*jIter)<N_nodes_input)       // from input layer
                {
                    X_t[i] += W_t[i*N_nodes_total+(*jIter)] * R_input[(*jIter)-0];
                }
                else if((*jIter)>=N_nodes_inout) // from hidden layer
                {
                    // X_t(j!=i)は逐次更新されており、Zeroにはなっていない前回の結果が入っている
                    X_t[i] += W_t[i*N_nodes_total+(*jIter)] * ConvThreshold(X_t[(*jIter)]);
                }
                else // output layer
                {
                    if(DoLearning)
                    {
                        X_t[i] += W_t[i*N_nodes_total+(*jIter)] * R_output[(*jIter)-N_nodes_input];
                    }
                    else
                    {
                        X_t[i] += W_t[i*N_nodes_total+(*jIter)] * ConvThreshold(X_t[(*jIter)]);
                    }
                }
            }
            // Innertia 
            X_t[i] = (1.-K_XupdateLambda) * X_tm1[i] + K_XupdateLambda * X_t[i];
            //
            if( (*jIter) >= N_nodes_inout ) // Hidden layer
            {
                // Firing with random imput
                R_t[i] = ConvThreshold( X_t[i] + rand_normal(0., K_rand_Xsigma) );
            }
            else // output layer
            {
                R_t[i] = ConvThreshold( X_t[i] );
            }
        }
        // Currently no connection to input is implemented

        // Learning only
        if(DoLearning)
        {
            // Fix the node status
            for(unsigned int i=0; i<N_nodes_input; ++i)
            {
                R_t[i] = R_input[i-0];
            }
            for(unsigned int i=N_nodes_input; i<N_nodes_inout; ++i)
            {
                R_t[i] = R_output[i-N_nodes_input];
            }

            // Update S
            for(unsigned int i=0; i<N_nodes_total; ++i)
            {
                S1[i] += R_t[i];
                S2[i*N_nodes_total+i] += R_t[i]*R_t[i];
            }
            // Update W
            for(unsigned int i=0; i<N_nodes_total; ++i)
            {
                for(jIter=Connection_to_from[i].begin();jIter!=Connection_to_from[i].end(); ++jIter)
                {
                    // W(i<-j)
                    // Key Question: ここは、X_tm1を使ってしまっているけど、R_tm1を使ったほうが良い？(この場合、学習が途中で止まる)
                    //それとも、X_tを使うほうが良い？
                    
                    // Key Question: あと、hidden layersに対しては別の学習則が必要

                    double deltaW             = K_update / ( N_iteration / S2[(*jIter)*N_nodes_total + (*jIter)] ) 
                                                / Connection_to_from[i].size() // Normalize by the number of connections
                                                //* R_t[(*jIter)] * ( R_t[i] - X_tm1[i] );
                                                * R_t[(*jIter)] * ( R_t[i] - X_t[i] );
                                                //* R_t[(*jIter)] * ( R_t[i] - ConvThreshold(X_t[i]) );
                    deltaW_int[i*N_nodes_total+(*jIter)] += deltaW; 
                    //if      ( W_t[i*N_nodes_total+(*jIter)] >= +1 ) W_t[i*N_nodes_total+(*jIter)]=+1;
                    //else if ( W_t[i*N_nodes_total+(*jIter)] <= -1 ) W_t[i*N_nodes_total+(*jIter)]=-1;
                }
            }
        }
        // Output
        DumpToFile(N_iteration, SampleNumber, iStep, comment);
    }
    return &(R_t[N_nodes_input]);
}
////////////////////////////////////////////
void NNN::ApplyDeltaW()
{
    for(unsigned int i=0; i<N_nodes_total; ++i)
    {
        for(vector< int >::iterator jIter=Connection_to_from[i].begin();jIter!=Connection_to_from[i].end(); ++jIter)
        {
            W_t[i*N_nodes_total+(*jIter)] += deltaW_int[i*N_nodes_total+*(jIter)]; 
            if      ( W_t[i*N_nodes_total+(*jIter)] >= +1 ) W_t[i*N_nodes_total+(*jIter)]=+1;
            else if ( W_t[i*N_nodes_total+(*jIter)] <= -1 ) W_t[i*N_nodes_total+(*jIter)]=-1;
        }
    }
    return;
}
////////////////////////////////////////////
NNN::NNN()
{
    p_Samples_input  = NULL;
    p_Samples_output = NULL;

    p_DumpList_X = new vector<unsigned int>;
    p_DumpList_R = new vector<unsigned int>;
    p_DumpList_W = new vector< pair<unsigned int, unsigned int> >;
}
////////////////////////////////////////////
NNN::~NNN()
{
    delete W_t;
    delete deltaW_int;
    delete R_t;
    delete R_tm1;
    delete X_t;
    delete X_tm1;
    delete S1;
    delete S2;
    delete p_DumpList_X;
    delete p_DumpList_R;
    delete p_DumpList_W;
    if(p_Samples_input!=NULL)
    {
        for(unsigned int i=0; i<p_Samples_input->size(); ++i)
        {
            delete (*p_Samples_input)[i];
        }
        delete p_Samples_input;
        p_Samples_input=NULL;
    }
    if(p_Samples_output!=NULL)
    {
        for(unsigned int i=0; i<p_Samples_output->size(); ++i)
        {
            delete (*p_Samples_output)[i];
        }
        delete p_Samples_output;
        p_Samples_output=NULL;
    }
}
////////////////////////////////////////////
////////////////////////////////////////////

int main()
{
    return 0;
}

extern "C" {
    PyObject* NNN_test(PyObject*, PyObject*);
    PyObject* NNN_AllocNeurons(PyObject*, PyObject*);
    PyObject* NNN_Initialize(PyObject*, PyObject*);
    PyObject* NNN_PrintX(PyObject*, PyObject*);
    PyObject* NNN_PrintR(PyObject*, PyObject*);
    PyObject* NNN_PrintS(PyObject*, PyObject*);
    PyObject* NNN_PrintW(PyObject*, PyObject*);
    PyObject* NNN_SetSamples(PyObject*, PyObject*);
    PyObject* NNN_SetConnection(PyObject*, PyObject*);
    PyObject* NNN_LearnOnce(PyObject*, PyObject*);
    PyObject* NNN_SetXsigma(PyObject*, PyObject*);
    PyObject* NNN_SetRecordTarget1D(PyObject*, PyObject*);
    PyObject* NNN_SetRecordTarget2D(PyObject*, PyObject*);
    PyObject* NNN_SetKupdateLambda(PyObject*, PyObject*);
    PyObject* NNN_ApplyDeltaW(PyObject*, PyObject*);
    void initNNN();
}

static NNN nnn;

//void test();
PyObject* NNN_test(PyObject* self, PyObject* args)
{
    nnn.test();
    return Py_BuildValue("");
}

//void Initialize(unsigned int m_N_nodes_input, unsigned int m_N_nodes_output, unsigned int m_N_nodes_hidden);
PyObject* NNN_AllocNeurons(PyObject* self, PyObject* args)
{
    unsigned int m_N_nodes_input, m_N_nodes_output, m_N_nodes_hidden;
    if (!PyArg_ParseTuple(args, "III", &m_N_nodes_input, &m_N_nodes_output, &m_N_nodes_hidden)) return NULL;
    nnn.AllocNeurons(m_N_nodes_input, m_N_nodes_output, m_N_nodes_hidden);
    return Py_BuildValue("");
}

//void SetConnection(unsigned int from, unsigned int to);
PyObject* NNN_SetConnection(PyObject* self, PyObject* args)
{
    unsigned int m_from, m_to;
    if (!PyArg_ParseTuple(args, "II", &m_from, &m_to)) return NULL;
    //cout<<"before: "<<m_from<<" "<<m_to<<endl;
    nnn.SetConnection(m_from, m_to);
    //cout<<"after : "<<m_from<<" "<<m_to<<endl;
    return Py_BuildValue("");
}
//void Initialize();
PyObject* NNN_Initialize(PyObject* self, PyObject* args)
{
    nnn.Initialize();
    return Py_BuildValue("");
}

//void PrintX(bool b_input=true, bool b_output=true, bool b_hidden=true);
PyObject* NNN_PrintX(PyObject* self, PyObject* args)
{
    int m_input, m_output, m_hidden;
    if (!PyArg_ParseTuple(args, "iii", &m_input, &m_output, &m_hidden)) return NULL;
    nnn.PrintX((bool)m_input, (bool)m_output, (bool)m_hidden);
    return Py_BuildValue("");
}

//void PrintR(bool b_input=true, bool b_output=true, bool b_hidden=true);
PyObject* NNN_PrintR(PyObject* self, PyObject* args)
{
    int m_input, m_output, m_hidden;
    if (!PyArg_ParseTuple(args, "iii", &m_input, &m_output, &m_hidden)) return NULL;
    nnn.PrintR((bool)m_input, (bool)m_output, (bool)m_hidden);
    return Py_BuildValue("");
}

//void PrintS(bool b_input=true, bool b_output=true, bool b_hidden=true);
PyObject* NNN_PrintS(PyObject* self, PyObject* args)
{
    int m_input, m_output, m_hidden;
    if (!PyArg_ParseTuple(args, "iii", &m_input, &m_output, &m_hidden)) return NULL;
    nnn.PrintS((bool)m_input, (bool)m_output, (bool)m_hidden);
    return Py_BuildValue("");
}

//void PrintW();
PyObject* NNN_PrintW(PyObject* self, PyObject* args)
{
    nnn.PrintW();
    return Py_BuildValue("");
}
//void SetXsigma();
PyObject* NNN_SetXsigma(PyObject* self, PyObject* args)
{
    double m_sigma;
    if (!PyArg_ParseTuple(args, "d", &m_sigma)) return NULL;
    nnn.SetXsigma(m_sigma);
    return Py_BuildValue("");
}
//void SetKupdate();
PyObject* NNN_SetKupdate(PyObject* self, PyObject* args)
{
    double m_Kupdate;
    if (!PyArg_ParseTuple(args, "d", &m_Kupdate)) return NULL;
    nnn.SetKupdate(m_Kupdate);
    return Py_BuildValue("");
}
//void SetupdateLambda();
PyObject* NNN_SetXupdateLambda(PyObject* self, PyObject* args)
{
    double m_K_XupdateLambda;
    if (!PyArg_ParseTuple(args, "d", &m_K_XupdateLambda)) return NULL;
    nnn.SetXupdateLambda(m_K_XupdateLambda);
    return Py_BuildValue("");
}

//void SetSamples();
PyObject* NNN_SetSamples(PyObject* self, PyObject* args)
{
    PyArrayObject *arrayX, *arrayY;
    PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &arrayX, &PyArray_Type, &arrayY);
    //npy_intp  ndimX  = PyArray_NDIM(arrayX);
    //if(ndim!=2) return NULL;
    npy_intp *ndimsX = PyArray_DIMS(arrayX);
    npy_intp *ndimsY = PyArray_DIMS(arrayY);
    vector< vector<int>* > *p_Samples_input  = new vector< vector<int>* >;
    vector< vector<int>* > *p_Samples_output = new vector< vector<int>* >;
    // Input
    p_Samples_input->resize(ndimsX[0]);
    //cout<<"ndimsX[0]="<<ndimsX[0]<<endl;
    //cout<<"ndimsX[1]="<<ndimsX[1]<<endl;
    for(unsigned int i=0; i<ndimsX[0]; ++i)
    {
        vector<int> *p_aSample=new vector<int>;
        p_aSample->resize(ndimsX[1]);
        for(unsigned int j=0; j<ndimsX[1]; ++j)
        {
            int a = *(int*)PyArray_GETPTR2(arrayX,i,j);
            (*p_aSample)[j]=a;
        }
        (*p_Samples_input)[i]=p_aSample;
    }
    // Output
    p_Samples_output->resize(ndimsY[0]);
    for(unsigned int i=0; i<ndimsY[0]; ++i)
    {
        vector<int> *p_aSample=new vector<int>;
        p_aSample->resize(ndimsY[1]);
        for(unsigned int j=0; j<ndimsY[1]; ++j)
        {
            int a = *(int*)PyArray_GETPTR2(arrayY,i,j);
            (*p_aSample)[j]=a;
        }
        (*p_Samples_output)[i]=p_aSample;
    }
    nnn.SetSamples(p_Samples_input, p_Samples_output);
    return Py_BuildValue("");
}

//int* LearnOnce(unsigned int SampleNumber  , unsigned int nSteps, bool DoInitialize=true, bool DoLearning = true);
PyObject* NNN_LearnOnce(PyObject* self, PyObject* args)
{
    unsigned int SampleNumber;
    unsigned int nSteps;
    char* Comment;
    int DoInitialize, DoLearning;
    if (!PyArg_ParseTuple(args, "IIsii", &SampleNumber, &nSteps, &Comment, &DoInitialize, &DoLearning)) return NULL;
    //string sComment = Comment;
    nnn.LearnOnce(SampleNumber, nSteps, Comment, (bool)DoInitialize, (bool)DoLearning);
    return Py_BuildValue("");
}
//void ApplyDeltaW();
PyObject* NNN_ApplyDeltaW(PyObject* self, PyObject* args)
{
    nnn.ApplyDeltaW();
    return Py_BuildValue("");
}
//void CloseRecord();
PyObject* NNN_CloseOutputfile(PyObject* self, PyObject* args)
{
    nnn.CloseOutputfile();
    return Py_BuildValue("");
}
//void OpenOutputfile(char* filename, bool CreateNewfile, bool m_DoDump_X, bool m_DoDump_R, bool m_DoDump_W);
PyObject* NNN_OpenOutputfile(PyObject* self, PyObject* args)
{
    char* filename;
    int   CreateNewfile;
    if (!PyArg_ParseTuple(args, "si", &filename, &CreateNewfile)) return NULL;
    nnn.OpenOutputfile(filename,(bool)CreateNewfile);
    return Py_BuildValue("");
}
//void SetRecordTarget1D(char* targetName, unsigned int tgtIdx);
PyObject* NNN_SetRecordTarget1D(PyObject* self, PyObject* args)
{
    char* targetName;
    unsigned int tgtIdx;
    if (!PyArg_ParseTuple(args, "sI", &targetName, &tgtIdx)) return NULL;
    nnn.SetRecordTarget1D(targetName, tgtIdx);
    return Py_BuildValue("");
}
//void SetRecordTarget2D(char* targetName, unsigned int from, unsigned int to);
PyObject* NNN_SetRecordTarget2D(PyObject* self, PyObject* args)
{
    char* targetName;
    unsigned int from,to;
    if (!PyArg_ParseTuple(args, "sII", &targetName, &from, &to)) return NULL;
    nnn.SetRecordTarget2D(targetName, from, to);
    return Py_BuildValue("");
}
static PyMethodDef NNNmethods[] = {
    {"test", NNN_test, METH_VARARGS},
    {"AllocNeurons", NNN_AllocNeurons, METH_VARARGS},
    {"Initialize", NNN_Initialize, METH_VARARGS},
    {"PrintX", NNN_PrintX, METH_VARARGS},
    {"PrintR", NNN_PrintR, METH_VARARGS},
    {"PrintS", NNN_PrintS, METH_VARARGS},
    {"PrintW", NNN_PrintW, METH_VARARGS},
    {"SetXsigma", NNN_SetXsigma, METH_VARARGS},
    {"SetKupdate", NNN_SetKupdate, METH_VARARGS},
    {"SetXupdateLambda", NNN_SetXupdateLambda, METH_VARARGS},
    {"SetSamples", NNN_SetSamples, METH_VARARGS},
    {"LearnOnce", NNN_LearnOnce, METH_VARARGS},
    {"SetConnection", NNN_SetConnection, METH_VARARGS},
    {"OpenOutputfile", NNN_OpenOutputfile, METH_VARARGS},
    {"CloseOutputfile", NNN_CloseOutputfile, METH_VARARGS},
    {"SetRecordTarget1D", NNN_SetRecordTarget1D, METH_VARARGS},
    {"SetRecordTarget2D", NNN_SetRecordTarget2D, METH_VARARGS},
    {"ApplyDeltaW", NNN_ApplyDeltaW, METH_VARARGS},
    {NULL},
};

void initNNN()
{
    Py_InitModule("NNN", NNNmethods);
    import_array();
}
/*
    int* LearnOnce(int *R_input, int *R_output, unsigned int nSteps, bool DoInitialize=true, bool DoLearning = true);
    double   K_update;
    double   K_rand_sigma;
    */
