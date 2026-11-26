/* Causal projection of one complete report stream. Whiteners have at most 14 rows.
   Invertible A makes the unknown state at each suffix start equivalent to one
   unknown state at window start. The suffix designs therefore accumulate. */
#include <math.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void scan(int paths, int ticks, int edges, int width, int rank, int begin,
                 int end, int startup, const double *reports, const double *keys,
                 const double *blocks, const double *ha, double *scores) {
    for (int p=0; p<paths; ++p) for (int t=begin; t<end; ++t) for (int e=0; e<edges; ++e) {
        double record[14]={0}, u[14]={0}, v[14]={0};
        for (int j=startup; j<width; ++j) {
            int step=((p*ticks+t-width+1+j)*edges+e);
            const double *w=blocks+((e*width+j)*rank)*2;
            const double *y=reports+step*2;
            for (int r=0; r<rank; ++r) record[r]+=w[2*r]*y[0]+w[2*r+1]*y[1];
        }
        double best=0, size_bound=0;
        for (int j=width-1; j>=startup; --j) {
            int step=((p*ticks+t-width+1+j)*edges+e);
            const double *w=blocks+((e*width+j)*rank)*2, *m=keys+step*4;
            const double *h=ha+(e*width+j)*4;
            double a=m[0]*h[0]+m[1]*h[2], b=m[0]*h[1]+m[1]*h[3];
            double c=m[2]*h[0]+m[3]*h[2], d=m[2]*h[1]+m[3]*h[3];
            double wnorm=0;
            for (int r=0;r<2*rank;++r) wnorm+=w[r]*w[r];
            size_bound+=sqrt(wnorm*(a*a+b*b+c*c+d*d));
            double uu=0,uv=0,vv=0,uy=0,vy=0;
            for(int r=0;r<rank;++r){
                u[r]+=w[2*r]*a+w[2*r+1]*c; v[r]+=w[2*r]*b+w[2*r+1]*d;
                uu+=u[r]*u[r]; uv+=u[r]*v[r]; vv+=v[r]*v[r];
                uy+=record[r]*u[r]; vy+=record[r]*v[r];
            }
            if (uu+vv==0) continue;
            const double *pivot=uu>=vv?u:v, *other=uu>=vv?v:u;
            double norm=fmax(uu,vv), first=uu>=vv?uy:vy;
            double coeff=uv/norm, perp=0, rhs=0;
            for(int r=0;r<rank;++r){
                double z=other[r]-coeff*pivot[r]; perp+=z*z; rhs+=record[r]*z;
            }
            double top=(uu+vv+hypot(uu-vv,2*uv))*.5;
            double floor2=1e-24*size_bound*size_bound;
            if(top<=floor2) continue;
            double score=first*first/norm;
            if(norm*perp>fmax(1e-20*top,floor2)*top) score+=rhs*rhs/perp;
            else {
                /* Dominant right singular vector; normalize in report space. */
                double angle=.5*atan2(2*uv,uu-vv);
                double component=cos(angle)*uy+sin(angle)*vy;
                score=component*component/top;
            }
            if(score>best) best=score;
        }
        scores[(p*ticks+t)*edges+e]=best;
    }
}
