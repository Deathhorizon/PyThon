#!/usr/bin/env python
'''
import execjs
default = execjs.get().name
print(default)
print(execjs.eval('const Qs = require("qs")'))'''
import execjs
ctx = execjs.compile("""
    function replace(val){
    const Qs = require('qs');
    let url = val;
    return JSON.stringify(Qs.parse(url));
    }
""")
print(execjs.get().name)
print(ctx.call('replace','hteId=&HteName=07-09+9%3A42+%E4%BD%9C%E4%B8%9A&Generatetype=Manually&HTEType=HomeWork&CourseId=4294971554648&UserId=1292785777203&IsPublishedNow=true&EndTimeModel=OneWeek&StartTime=Tue+Jul+09+2019+09%3A42%3A28+GMT%2B0800+(%E4%B8%AD%E5%9B%BD%E6%A0%87%E5%87%86%E6%97%B6%E9%97%B4)&EndTime=Tue+Jul+16+2019+09%3A42%3A28+GMT%2B0800+(%E4%B8%AD%E5%9B%BD%E6%A0%87%E5%87%86%E6%97%B6%E9%97%B4)&Exercises%5B0%5D%5BExerciseId%5D=3006521958394&Exercises%5B0%5D%5BScore%5D=1&Exercises%5B1%5D%5BExerciseId%5D=3006521919767&Exercises%5B1%5D%5BScore%5D=1&ResSearchExercises%5B0%5D%5BExerciseId%5D=3006521958394&ResSearchExercises%5B0%5D%5BET%5D=Choice&ResSearchExercises%5B0%5D%5BScore%5D=1&ResSearchExercises%5B0%5D%5BExerciseSubject%5D=%25E6%259F%2590%25E9%259C%25B2%25E5%25A4%25A9%25E8%2588%259E%25E5%258F%25B0%25E5%25A6%2582%25E5%259B%25BE%25E6%2589%2580%25E7%25A4%25BA%25EF%25BC%258C%25E5%25AE%2583%25E7%259A%2584%25E4%25BF%25AF%25E8%25A7%2586%25E5%259B%25BE%25E6%2598%25AF%25EF%25BC%2588%2520%2526nbsp%253B%2526nbsp%253B%25EF%25BC%2589%2520%2520%253Cp%2520align%253D%2522left%2522%253E%2520%253Cimg%2520width%253D%2522178%2522%2520height%253D%252280%2522%2520src%253D%2522https%253A%252F%252Ftikupic.21cnjy.com%252F46%252F03%252F4603e0f2276c422f3bdcbf7a38acc2ea.png%2522%2520v%253Ashapes%253D%2522%25E5%259B%25BE%25E7%2589%2587_x0020_4%2522%253E%2520%253C%252Fp%253E%2520%2520%2520&ResSearchExercises%5B0%5D%5BItemIndex%5D=1&ResSearchExercises%5B0%5D%5BExerciseTypeAliasID%5D=863288426577&ResSearchExercises%5B0%5D%5BExerciseType%5D=%E5%8D%95%E9%80%89%E9%A2%98&ResSearchExercises%5B0%5D%5BAnswersOptions%5D%5B%5D=%253Cimg%2520width%253D%2522157%2522%2520height%253D%252231%2522%2520src%253D%2522https%253A%252F%252Ftikupic.21cnjy.com%252F29%252F8d%252F298d918e89016d05b6d1972ef0fb83fc.png%2522%2520v%253Ashapes%253D%2522%25E5%259B%25BE%25E7%2589%2587_x0020_27%2522%253E&ResSearchExercises%5B0%5D%5BAnswersOptions%5D%5B%5D=%253Cimg%2520width%253D%2522158%2522%2520height%253D%252286%2522%2520src%253D%2522https%253A%252F%252Ftikupic.21cnjy.com%252Fbe%252F4c%252Fbe4c0f91a74e0496b57fc8a19a00ba2b.png%2522%2520v%253Ashapes%253D%2522%25E5%259B%25BE%25E7%2589%2587_x0020_28%2522%253E%2520%2520%2520&ResSearchExercises%5B0%5D%5BAnswersOptions%5D%5B%5D=%253Cimg%2520width%253D%252299%2522%2520height%253D%252231%2522%2520src%253D%2522https%253A%252F%252Ftikupic.21cnjy.com%252F46%252Fd0%252F46d0e4d754a5a113cf7dd225774bb311.png%2522%2520v%253Ashapes%253D%2522%25E5%259B%25BE%25E7%2589%2587_x0020_29%2522%253E&ResSearchExercises%5B0%5D%5BAnswersOptions%5D%5B%5D=%253Cimg%2520width%253D%252287%2522%2520height%253D%252231%2522%2520src%253D%2522https%253A%252F%252Ftikupic.21cnjy.com%252Fc7%252F8c%252Fc78cb8542e2246afb196cf1bb4539683.png%2522%2520v%253Ashapes%253D%2522%25E5%259B%25BE%25E7%2589%2587_x0020_30%2522%253E%2520%2520%2520&ResSearchExercises%5B0%5D%5BextendInfo%5D%5BExtend_ExerciseID%5D=11698906&ResSearchExercises%5B0%5D%5BextendInfo%5D%5Bstage%5D=0&ResSearchExercises%5B0%5D%5BextendInfo%5D%5Bsubject%5D=0&ResSearchExercises%5B0%5D%5BextendInfo%5D%5Btype%5D=1&ResSearchExercises%5B0%5D%5BextendInfo%5D%5Bbasetype%5D=1&ResSearchExercises%5B0%5D%5BextendInfo%5D%5BIFSingleChoice%5D=true&ResSearchExercises%5B0%5D%5BExtend_ExerciseID%5D=11698906&ResSearchExercises%5B0%5D%5BIsExtendExercise%5D=true&ResSearchExercises%5B0%5D%5BIsSingleChoice%5D=1&ResSearchExercises%5B1%5D%5BExerciseId%5D=3006521919767&ResSearchExercises%5B1%5D%5BET%5D=Choice&ResSearchExercises%5B1%5D%5BScore%5D=1&ResSearchExercises%5B1%5D%5BExerciseSubject%5D=%25E5%25A6%2582%25E5%259B%25BE%25EF%25BC%258C%25E5%259C%25A8%25E7%259F%25A9%25E5%25BD%25A2ABCD%25E4%25B8%25AD%25EF%25BC%258CE%25E4%25B8%25BAAB%25E4%25B8%25AD%25E7%2582%25B9%25EF%25BC%258C%25E4%25BB%25A5BE%25E4%25B8%25BA%25E8%25BE%25B9%25E4%25BD%259C%25E6%25AD%25A3%25E6%2596%25B9%25E5%25BD%25A2BEFG%25EF%25BC%258C%25E8%25BE%25B9EF%25E4%25BA%25A4CD%25E4%25BA%258E%25E7%2582%25B9H%25EF%25BC%258C%25E5%259C%25A8%25E8%25BE%25B9BE%25E4%25B8%258A%25E5%258F%2596%25E7%2582%25B9M%25E4%25BD%25BFBM%25EF%25BC%259DBC%25EF%25BC%258C%25E4%25BD%259CMN%25E2%2588%25A5BG%25E4%25BA%25A4CD%25E4%25BA%258E%25E7%2582%25B9L%25EF%25BC%258C%25E4%25BA%25A4FG%25E4%25BA%258E%25E7%2582%25B9N%25EF%25BC%258E%25E6%25AC%25A7%25E5%2584%25BF%25E9%2587%258C%25E5%25BE%2597%25E5%259C%25A8%25E3%2580%258A%25E5%2587%25A0%25E4%25BD%2595%25E5%258E%259F%25E6%259C%25AC%25E3%2580%258B%25E4%25B8%25AD%25E5%2588%25A9%25E7%2594%25A8%25E8%25AF%25A5%25E5%259B%25BE%25E8%25A7%25A3%25E9%2587%258A%25E4%25BA%2586%2520%253Cimg%2520class%253D%2522mathml%2522%2520src%253D%2522http%253A%252F%252Fmath.21cnjy.com%252FMathMLToImage%253Fmml%253D%25253Cmath%252Bdisplay%25253D%252522block%252522%252Bxmlns%25253D%252522http%25253A%25252F%25252Fwww.w3.org%25252F1998%25252FMath%25252FMathML%252522%25253E%25253Cmrow%25253E%25253Cmo%252Bstretchy%25253D%252522false%252522%25253E%252528%25253C%25252Fmo%25253E%25253Cmi%25253Ea%25253C%25252Fmi%25253E%25253Cmo%25253E%25252B%25253C%25252Fmo%25253E%25253Cmi%25253Eb%25253C%25252Fmi%25253E%25253Cmo%252Bstretchy%25253D%252522false%252522%25253E%252529%25253C%25252Fmo%25253E%25253Cmo%252Bstretchy%25253D%252522false%252522%25253E%252528%25253C%25252Fmo%25253E%25253Cmi%25253Ea%25253C%25252Fmi%25253E%25253Cmo%25253E%2525E2%252588%252592%25253C%25252Fmo%25253E%25253Cmi%25253Eb%25253C%25252Fmi%25253E%25253Cmo%252Bstretchy%25253D%252522false%252522%25253E%252529%25253C%25252Fmo%25253E%25253Cmo%25253E%25253D%25253C%25252Fmo%25253E%25253Cmsup%25253E%25253Cmi%25253Ea%25253C%25252Fmi%25253E%25253Cmn%25253E2%25253C%25252Fmn%25253E%25253C%25252Fmsup%25253E%25253Cmo%25253E%2525E2%252588%252592%25253C%25252Fmo%25253E%25253Cmsup%25253E%25253Cmi%25253Eb%25253C%25252Fmi%25253E%25253Cmn%25253E2%25253C%25252Fmn%25253E%25253C%25252Fmsup%25253E%25253C%25252Fmrow%25253E%25253C%25252Fmath%25253E%2526key%253D4879004862d17115f5e110593a6f860c%2522%2520style%253D%2522vertical-align%253A%2520middle%253B%2522%2520%252F%253E%2520%25EF%25BC%258E%25E7%258E%25B0%25E4%25BB%25A5%25E7%2582%25B9F%25E4%25B8%25BA%25E5%259C%2586%25E5%25BF%2583%25EF%25BC%258CFE%25E4%25B8%25BA%25E5%258D%258A%25E5%25BE%2584%25E4%25BD%259C%25E5%259C%2586%25E5%25BC%25A7%25E4%25BA%25A4%25E7%25BA%25BF%25E6%25AE%25B5DH%25E4%25BA%258E%25E7%2582%25B9P%25EF%25BC%258C%25E8%25BF%259E%25E7%25BB%2593EP%25EF%25BC%258C%25E8%25AE%25B0%25E2%2596%25B3EPH%25E7%259A%2584%25E9%259D%25A2%25E7%25A7%25AF%25E4%25B8%25BAS%253Csub%253E1%253C%252Fsub%253E%25EF%25BC%258C%25E5%259B%25BE%25E4%25B8%25AD%25E9%2598%25B4%25E5%25BD%25B1%25E9%2583%25A8%25E5%2588%2586%25E7%259A%2584%25E9%259D%25A2%25E7%25A7%25AF%25E4%25B8%25BAS%253Csub%253E2%253C%252Fsub%253E%25EF%25BC%258E%25E8%258B%25A5%25E7%2582%25B9A%25EF%25BC%258CL%25EF%25BC%258CG%25E5%259C%25A8%25E5%2590%258C%25E4%25B8%2580%25E7%259B%25B4%25E7%25BA%25BF%25E4%25B8%258A%25EF%25BC%258C%25E5%2588%2599%2520%253Cimg%2520class%253D%2522mathml%2522%2520src%253D%2522http%253A%252F%252Fmath.21cnjy.com%252FMathMLToImage%253Fmml%253D%25253Cmath%252Bdisplay%25253D%252522block%252522%252Bxmlns%25253D%252522http%25253A%25252F%25252Fwww.w3.org%25252F1998%25252FMath%25252FMathML%252522%25253E%25253Cmrow%25253E%25253Cmfrac%25253E%25253Cmrow%25253E%25253Cmsub%25253E%25253Cmtext%25253ES%25253C%25252Fmtext%25253E%25253Cmn%25253E1%25253C%25252Fmn%25253E%25253C%25252Fmsub%25253E%25253C%25252Fmrow%25253E%25253Cmrow%25253E%25253Cmsub%25253E%25253Cmtext%25253ES%25253C%25252Fmtext%25253E%25253Cmn%25253E2%25253C%25252Fmn%25253E%25253C%25252Fmsub%25253E%25253C%25252Fmrow%25253E%25253C%25252Fmfrac%25253E%25253C%25252Fmrow%25253E%25253C%25252Fmath%25253E%2526key%253D4879004862d17115f5e110593a6f860c%2522%2520style%253D%2522vertical-align%253A%2520middle%253B%2522%2520%252F%253E%2520%25E7%259A%2584%25E5%2580%25BC%25E4%25B8%25BA%25EF%25BC%2588%2520%2526nbsp%253B%2526nbsp%253B%25EF%25BC%2589%2520%2520%253Cp%2520align%253D%2522left%2522%253E%2520%253Cimg%2520width%253D%2522249%2522%2520height%253D%2522149%2522%2520src%253D%2522https%253A%252F%252Ftikupic.21cnjy.com%252F18%252Fec%252F18ec91e810fabd915e5b9e6c41273ee1.png%2522%2520v%253Ashapes%253D%2522%25E5%259B%25BE%25E7%2589%2587_x0020_32%2522%253E%2520%253C%252Fp%253E%2520%2520%2520&ResSearchExercises%5B1%5D%5BItemIndex%5D=2&ResSearchExercises%5B1%5D%5BExerciseTypeAliasID%5D=863288426577&ResSearchExercises%5B1%5D%5BExerciseType%5D=%E5%8D%95%E9%80%89%E9%A2%98&ResSearchExercises%5B1%5D%5BAnswersOptions%5D%5B%5D=%253Cimg%2520class%253D%2522mathml%2522%2520src%253D%2522http%253A%252F%252Fmath.21cnjy.com%252FMathMLToImage%253Fmml%253D%25253Cmath%252Bdisplay%25253D%252522block%252522%252Bxmlns%25253D%252522http%25253A%25252F%25252Fwww.w3.org%25252F1998%25252FMath%25252FMathML%252522%25253E%25253Cmrow%25253E%25253Cmfrac%25253E%25253Cmrow%25253E%25253Cmsqrt%25253E%25253Cmn%25253E2%25253C%25252Fmn%25253E%25253C%25252Fmsqrt%25253E%25253C%25252Fmrow%25253E%25253Cmn%25253E2%25253C%25252Fmn%25253E%25253C%25252Fmfrac%25253E%25253C%25252Fmrow%25253E%25253C%25252Fmath%25253E%2526key%253D4879004862d17115f5e110593a6f860c%2522%2520style%253D%2522vertical-align%253A%2520middle%253B%2522%2520%252F%253E&ResSearchExercises%5B1%5D%5BAnswersOptions%5D%5B%5D=%253Cimg%2520class%253D%2522mathml%2522%2520src%253D%2522http%253A%252F%252Fmath.21cnjy.com%252FMathMLToImage%253Fmml%253D%25253Cmath%252Bdisplay%25253D%252522block%252522%252Bxmlns%25253D%252522http%25253A%25252F%25252Fwww.w3.org%25252F1998%25252FMath%25252FMathML%252522%25253E%25253Cmrow%25253E%25253Cmfrac%25253E%25253Cmrow%25253E%25253Cmsqrt%25253E%25253Cmn%25253E2%25253C%25252Fmn%25253E%25253C%25252Fmsqrt%25253E%25253C%25252Fmrow%25253E%25253Cmn%25253E3%25253C%25252Fmn%25253E%25253C%25252Fmfrac%25253E%25253C%25252Fmrow%25253E%25253C%25252Fmath%25253E%2526key%253D4879004862d17115f5e110593a6f860c%2522%2520style%253D%2522vertical-align%253A%2520middle%253B%2522%2520%252F%253E&ResSearchExercises%5B1%5D%5BAnswersOptions%5D%5B%5D=%253Cimg%2520class%253D%2522mathml%2522%2520src%253D%2522http%253A%252F%252Fmath.21cnjy.com%252FMathMLToImage%253Fmml%253D%25253Cmath%252Bdisplay%25253D%252522block%252522%252Bxmlns%25253D%252522http%25253A%25252F%25252Fwww.w3.org%25252F1998%25252FMath%25252FMathML%252522%25253E%25253Cmrow%25253E%25253Cmfrac%25253E%25253Cmrow%25253E%25253Cmsqrt%25253E%25253Cmn%25253E2%25253C%25252Fmn%25253E%25253C%25252Fmsqrt%25253E%25253C%25252Fmrow%25253E%25253Cmn%25253E4%25253C%25252Fmn%25253E%25253C%25252Fmfrac%25253E%25253C%25252Fmrow%25253E%25253C%25252Fmath%25253E%2526key%253D4879004862d17115f5e110593a6f860c%2522%2520style%253D%2522vertical-align%253A%2520middle%253B%2522%2520%252F%253E&ResSearchExercises%5B1%5D%5BAnswersOptions%5D%5B%5D=%253Cimg%2520class%253D%2522mathml%2522%2520src%253D%2522http%253A%252F%252Fmath.21cnjy.com%252FMathMLToImage%253Fmml%253D%25253Cmath%252Bdisplay%25253D%252522block%252522%252Bxmlns%25253D%252522http%25253A%25252F%25252Fwww.w3.org%25252F1998%25252FMath%25252FMathML%252522%25253E%25253Cmrow%25253E%25253Cmfrac%25253E%25253Cmrow%25253E%25253Cmsqrt%25253E%25253Cmn%25253E2%25253C%25252Fmn%25253E%25253C%25252Fmsqrt%25253E%25253C%25252Fmrow%25253E%25253Cmn%25253E6%25253C%25252Fmn%25253E%25253C%25252Fmfrac%25253E%25253C%25252Fmrow%25253E%25253C%25252Fmath%25253E%2526key%253D4879004862d17115f5e110593a6f860c%2522%2520style%253D%2522vertical-align%253A%2520middle%253B%2522%2520%252F%253E%2520%2520%2520&ResSearchExercises%5B1%5D%5BextendInfo%5D%5BExtend_ExerciseID%5D=11698920&ResSearchExercises%5B1%5D%5BextendInfo%5D%5Bstage%5D=0&ResSearchExercises%5B1%5D%5BextendInfo%5D%5Bsubject%5D=0&ResSearchExercises%5B1%5D%5BextendInfo%5D%5Btype%5D=1&ResSearchExercises%5B1%5D%5BextendInfo%5D%5Bbasetype%5D=1&ResSearchExercises%5B1%5D%5BextendInfo%5D%5BIFSingleChoice%5D=true&ResSearchExercises%5B1%5D%5BExtend_ExerciseID%5D=11698920&ResSearchExercises%5B1%5D%5BIsExtendExercise%5D=true&ResSearchExercises%5B1%5D%5BIsSingleChoice%5D=1&classIds%5B%5D=4294971574036&classIds%5B%5D=4294971574037&classIds%5B%5D=4294971574038&classIds%5B%5D=4294971578788&classIds%5B%5D=4294971614738&classNames%5B%5D=01%E7%8F%AD&classNames%5B%5D=%E8%AF%95%E7%94%A8%E7%8F%AD&classNames%5B%5D=3.14%E7%8F%AD2&classNames%5B%5D=mapi%E7%8F%AD&classNames%5B%5D=%E4%B8%AA%E4%BA%BA%E4%B8%93%E7%94%A8'))

