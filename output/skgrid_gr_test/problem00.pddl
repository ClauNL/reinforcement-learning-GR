
(define (problem p024-microban-sequential) (:domain leogrid)
  (:objects
        dir-down - direction
	dir-left - direction
	dir-right - direction
	dir-up - direction
	player-01 - thing
	pos-1-1 - location
	pos-1-2 - location
	pos-1-3 - location
	pos-1-4 - location
	pos-2-1 - location
	pos-2-2 - location
	pos-2-3 - location
	pos-2-4 - location
	pos-3-1 - location
	pos-3-2 - location
	pos-3-3 - location
	pos-3-4 - location
	pos-4-1 - location
	pos-4-2 - location
	pos-4-3 - location
	pos-4-4 - location
  )
  (:goal (and
	(at player-01 pos-4-4)
  ))
  (:init 
	(at player-01 pos-1-1)

	(clear pos-1-2)
	(clear pos-1-3)
	(clear pos-1-4)
	(clear pos-2-1)
	(clear pos-2-2)
	(clear pos-2-3)
	(clear pos-2-4)
	(clear pos-3-1)
	(clear pos-3-2)
	(clear pos-3-3)
	(clear pos-3-4)
	(clear pos-4-1)
	(clear pos-4-2)
	(clear pos-4-3)
	(clear pos-4-4)

	
	(is-player player-01)

	(move dir-down)
	(move dir-left)
	(move dir-right)
	(move dir-up)
	(move-dir pos-1-1 pos-2-1 dir-right)
	(move-dir pos-1-1 pos-1-2 dir-down)

	(move-dir pos-1-2 pos-1-1 dir-up)
	(move-dir pos-1-2 pos-1-3 dir-down)
	(move-dir pos-1-2 pos-2-2 dir-right)

	(move-dir pos-1-3 pos-1-2 dir-up)
	(move-dir pos-1-3 pos-1-4 dir-down)
	(move-dir pos-1-3 pos-2-3 dir-right)

	(move-dir pos-1-4 pos-1-3 dir-up)
	(move-dir pos-1-4 pos-2-4 dir-right)


	(move-dir pos-2-1 pos-2-2 dir-down)
	(move-dir pos-2-1 pos-1-1 dir-left)
	(move-dir pos-2-1 pos-3-1 dir-right)

	(move-dir pos-2-2 pos-2-1 dir-up)
	(move-dir pos-2-2 pos-2-3 dir-down)
	(move-dir pos-2-2 pos-1-2 dir-left)
	(move-dir pos-2-2 pos-3-2 dir-right)

	(move-dir pos-2-3 pos-2-2 dir-up)
	(move-dir pos-2-3 pos-2-4 dir-down)
	(move-dir pos-2-3 pos-1-3 dir-left)
	(move-dir pos-2-3 pos-3-3 dir-right)

	(move-dir pos-2-4 pos-2-3 dir-up)
	(move-dir pos-2-4 pos-1-4 dir-left)
	(move-dir pos-2-4 pos-3-4 dir-right)
	
	


	(move-dir pos-3-1 pos-2-1 dir-left)
	(move-dir pos-3-1 pos-3-2 dir-down)
	(move-dir pos-3-1 pos-4-1 dir-right)

	(move-dir pos-3-2 pos-3-1 dir-up)
	(move-dir pos-3-2 pos-3-3 dir-down)
	(move-dir pos-3-2 pos-2-2 dir-left)
	(move-dir pos-3-2 pos-4-2 dir-right)

	(move-dir pos-3-3 pos-3-2 dir-up)
	(move-dir pos-3-3 pos-3-4 dir-down)
	(move-dir pos-3-3 pos-2-3 dir-left)
	(move-dir pos-3-3 pos-4-3 dir-right)

	(move-dir pos-3-4 pos-3-3 dir-up)
	(move-dir pos-3-4 pos-2-4 dir-left)
	(move-dir pos-3-4 pos-4-4 dir-right)





	(move-dir pos-4-1 pos-4-2 dir-down)
	(move-dir pos-4-1 pos-3-1 dir-left)
	
	(move-dir pos-4-2 pos-4-1 dir-up)
	(move-dir pos-4-2 pos-4-3 dir-down)
	(move-dir pos-4-2 pos-3-2 dir-left)

	(move-dir pos-4-3 pos-4-2 dir-up)
	(move-dir pos-4-3 pos-4-4 dir-down)
	(move-dir pos-4-3 pos-3-3 dir-left)

	(move-dir pos-4-4 pos-3-4 dir-left)
	(move-dir pos-4-4 pos-4-3 dir-up)




))
        