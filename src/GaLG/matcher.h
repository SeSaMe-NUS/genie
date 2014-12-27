#ifndef GaLG_matcher_h
#define GaLG_matcher_h

namespace GaLG {
  struct dim_matcher {
    virtual bool match(int value) = 0;
    virtual ~dim_matcher(){};
  };

  namespace matcher {
    struct unset : dim_matcher {
      bool match(int value);
      ~unset(){};
    };

    struct range : dim_matcher {
      int _low, _up;
      range(int low, int up);
      bool match(int value);
      ~range(){};
    };
  }
}

#endif